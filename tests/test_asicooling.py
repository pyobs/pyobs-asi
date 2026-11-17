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
        self.power = 30
        self.writes: list[tuple[int, int]] = []

    def get_control_value(self, control: int) -> tuple[int, bool]:
        if control == asi.ASI_COOLER_ON:
            return int(self.cooler_on), False
        if control == asi.ASI_TARGET_TEMP:
            return self.target, False
        if control == asi.ASI_TEMPERATURE:
            return int(round(self.sensor * 10)), False
        if control == asi.ASI_COOLER_POWER_PERC:
            return self.power, False
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


def test_ramper_keeps_warming_while_the_sensor_lags_and_the_cooler_works() -> None:
    # sensor 5 degC below the setpoint, but the cooler still gives power: switching off now would warm up abruptly
    camera = FakeCamera(sensor=-25.0, cooler_on=True, target=-20, offset=5.0)
    assert SdkCoolingRamper(camera).step(False, 0) is not None
    assert camera.cooler_on is True


def test_ramper_switches_off_when_the_cooler_is_idle() -> None:
    camera = FakeCamera(sensor=-25.0, cooler_on=True, target=-20, offset=5.0)
    camera.power = 0
    assert SdkCoolingRamper(camera).step(False, 0) is None
    assert camera.cooler_on is False


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


def test_warmup_minutes_follows_warm_rate() -> None:
    camera = FakeCamera(sensor=-19.5, cooler_on=True, target=-20)
    assert SdkCoolingRamper(camera).warmup_minutes() == 40.0  # -20 -> 20 degC at 1 degC/min


def test_warmup_minutes_is_none_when_cooler_is_off() -> None:
    camera = FakeCamera(sensor=5.0, cooler_on=False)
    assert SdkCoolingRamper(camera).warmup_minutes() is None


def test_module_warms_up_on_close_by_default() -> None:
    module = AsiCoolCamera(camera="test")
    assert module._cooling_warmup_on_close is True
    # without a given timeout the core mixin estimates it from the setpoint when closing
    assert module._cooling_close_timeout is None


def test_module_warm_up_defaults() -> None:
    module = AsiCoolCamera(camera="test")
    assert module._cooling_ramp.off_temperature == 15.0
    assert module._cooling_ramp.rate_warm == 2.0


def test_module_warm_up_can_be_configured() -> None:
    module = AsiCoolCamera(camera="test", cooling_off_temperature=15.0, cooling_rate_warm=1.0)
    assert module._cooling_ramp.off_temperature == 15.0
    assert module._cooling_ramp.rate_warm == 1.0


def test_module_warmup_on_close_can_be_disabled() -> None:
    module = AsiCoolCamera(camera="test", cooling_warmup_on_close=False, cooling_close_timeout=10.0)
    assert module._cooling_warmup_on_close is False
    assert module._cooling_close_timeout == 10.0


@pytest.mark.asyncio
async def test_open_waits_for_the_first_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    module = AsiCoolCamera(camera="test")
    module._get_temperature = AsyncMock(side_effect=[0.0, 0.0, 17.6])  # type: ignore[method-assign]
    monkeypatch.setattr("pyobs_asi.asicamera._TEMPERATURE_READY_POLL", 0.0)

    await module._wait_for_temperature()

    assert module._get_temperature.await_count == 3  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_open_accepts_a_temperature_of_zero_after_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = AsiCoolCamera(camera="test")
    module._get_temperature = AsyncMock(return_value=0.0)  # type: ignore[method-assign]
    monkeypatch.setattr("pyobs_asi.asicamera._TEMPERATURE_READY_TIMEOUT", 0.05)
    monkeypatch.setattr("pyobs_asi.asicamera._TEMPERATURE_READY_POLL", 0.01)

    await module._wait_for_temperature()  # returns, the sensor may really be at 0 degC

    assert module._get_temperature.await_count >= 2  # type: ignore[attr-defined]
