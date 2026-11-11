"""Tests for the cooling ramp of AsiCoolCamera and of the SDK ramper used by the GUI, with a fake camera."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import zwoasi as asi  # type: ignore
from pyobs.interfaces import ICooling

from pyobs_asi import AsiCoolCamera
from pyobs_asi.coolingramper import SdkCoolingRamper


class FakeCamera:
    """Stands in for a zwoasi camera. The sensor follows the target setpoint at a fixed offset."""

    def __init__(self, sensor: float, cooler_on: bool, target: int = 0, offset: float = 0.5) -> None:
        self.cooler_on = cooler_on
        self.target = target
        self.sensor = sensor
        self.offset = offset
        self.writes: list[tuple[int, int]] = []

    def get_control_value(self, control: int) -> tuple[int, bool]:
        if control == asi.ASI_COOLER_ON:
            return int(self.cooler_on), False
        if control == asi.ASI_TARGET_TEMP:
            return self.target, False
        if control == asi.ASI_TEMPERATURE:
            return int(round(self.sensor * 10)), False
        if control == asi.ASI_COOLER_POWER_PERC:
            return 30, False
        raise AssertionError(control)

    def set_control_value(self, control: int, value: int) -> None:
        self.writes.append((control, value))
        if control == asi.ASI_COOLER_ON:
            self.cooler_on = bool(value)
        elif control == asi.ASI_TARGET_TEMP:
            self.target = value
            if self.cooler_on:
                self.sensor = value - self.offset


def run(ramper: SdkCoolingRamper, enabled: bool, set_point: float) -> int:
    steps = 0
    while ramper.step(enabled, set_point) is not None:
        steps += 1
        assert steps < 200
    return steps


def test_ramper_cools_down_in_steps_from_current_setpoint() -> None:
    camera = FakeCamera(sensor=-9.5, cooler_on=True, target=-10)
    run(SdkCoolingRamper(camera), True, -13)
    assert [v for c, v in camera.writes if c == asi.ASI_TARGET_TEMP] == [-11, -12, -13]


def test_ramper_starts_at_sensor_temperature_when_cooler_is_off() -> None:
    camera = FakeCamera(sensor=5.0, cooler_on=False)
    run(SdkCoolingRamper(camera), True, 3)
    assert [v for c, v in camera.writes if c == asi.ASI_TARGET_TEMP][:2] == [5, 4]
    assert camera.cooler_on and camera.target == 3


def test_ramper_warms_up_before_switching_off() -> None:
    camera = FakeCamera(sensor=-19.5, cooler_on=True, target=-20)
    run(SdkCoolingRamper(camera), False, 0)
    assert camera.cooler_on is False
    # setpoint was raised step by step first
    assert [v for c, v in camera.writes if c == asi.ASI_TARGET_TEMP][:3] == [-19, -18, -17]


def test_ramper_does_nothing_when_already_off() -> None:
    camera = FakeCamera(sensor=10.0, cooler_on=False)
    assert SdkCoolingRamper(camera).step(False, 0) is None
    assert camera.writes == []


def make_cool_camera() -> tuple[AsiCoolCamera, FakeCamera]:
    module = AsiCoolCamera(camera="test")
    module.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    fake = FakeCamera(sensor=-9.5, cooler_on=True, target=-10)
    module._camera = fake  # type: ignore[assignment]
    return module, fake


@pytest.mark.asyncio
async def test_cool_camera_set_cooling_queues_ramp() -> None:
    module, fake = make_cool_camera()
    await module.set_cooling(True, -20.0)
    assert module.cooling_ramping
    assert fake.writes == []


@pytest.mark.asyncio
async def test_cool_camera_ticks_write_whole_degrees_and_publish_target() -> None:
    module, fake = make_cool_camera()
    await module.set_cooling(True, -12.0)

    while module.cooling_ramping:
        await module._cooling_tick()

    assert [v for c, v in fake.writes if c == asi.ASI_TARGET_TEMP] == [-11, -12]
    # the state published with the last write still says ramping, the next poll reports the final state
    await module._publish_cooling()
    set_state: AsyncMock = module.comm.set_state  # type: ignore[assignment]
    interface, state = set_state.await_args.args  # type: ignore[union-attr]
    assert interface is ICooling
    assert state.setpoint == -12.0 and state.ramping is False


@pytest.mark.asyncio
async def test_cool_camera_disable_ramps_up_then_switches_off() -> None:
    module, fake = make_cool_camera()
    await module.set_cooling(False, 0.0)

    for _ in range(100):
        if not module.cooling_ramping:
            break
        await module._cooling_tick()

    assert fake.cooler_on is False
    assert not module.cooling_ramping
