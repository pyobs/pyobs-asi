"""Unit tests for the non-hardware logic in AsiCamera: constructor defaults, the
state-setting methods (window/binning/gain/offset/image format), and the
_run_blocking/_run_blocking_or_raise thread wrappers.

Hardware I/O (opening a camera, exposing, reading out) is out of scope here.
"""

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pyobs.events import ModeChangedEvent
from pyobs.interfaces import IBinning, IGain, IMode, IWindow
from pyobs.utils import exceptions as exc
from pyobs.utils.enums import ImageFormat

from pyobs_asi import AsiCamera, AsiCoolCamera
from pyobs_asi.presets import GAIN_GROUP, PRESETS, detector_gain

ASI2600 = "ZWO ASI2600MM Pro"


def _published(camera: AsiCamera, interface: type) -> Any:
    """State last published for an interface."""
    calls = [c.args for c in camera.comm.set_state.await_args_list]  # type: ignore[attr-defined]
    return [state for iface, state in calls if iface is interface][-1]


def test_constructor_defaults() -> None:
    camera = AsiCamera(camera="test")
    assert camera._camera_name == "test"
    assert camera._sdk_path == "/usr/local/lib/libASICamera2.so"
    assert camera._camera is None
    assert camera._window == (0, 0, 0, 0)
    assert camera._binning == 1
    assert camera._image_format == ImageFormat.INT16
    assert camera._gain == 1.0
    assert camera._gain_offset == 50.0


def test_cool_camera_default_setpoint() -> None:
    camera = AsiCoolCamera(camera="test")
    assert camera._temp_setpoint == -20


@pytest.mark.asyncio
async def test_set_window() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_window(10, 20, 100, 200)

    assert camera._window == (10, 20, 100, 200)
    assert camera.comm.set_state.await_args is not None
    interface, state = camera.comm.set_state.await_args.args
    assert interface is IWindow
    assert (state.x, state.y, state.width, state.height) == (10, 20, 100, 200)


@pytest.mark.asyncio
async def test_set_binning() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_binning(2, 3)

    # only x is tracked as the scalar binning factor; both axes still go out in the state
    assert camera._binning == 2
    assert camera.comm.set_state.await_args is not None
    interface, state = camera.comm.set_state.await_args.args
    assert interface is IBinning
    assert (state.x, state.y) == (2, 3)


@pytest.mark.asyncio
async def test_set_gain() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_gain(5.0)

    # offset is untouched, but the published state still carries the last-known offset
    assert camera._gain == 5.0
    state = _published(camera, IGain)
    assert (state.gain, state.offset) == (5.0, camera._gain_offset)


@pytest.mark.asyncio
async def test_set_offset() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_offset(25.0)

    assert camera._gain_offset == 25.0
    state = _published(camera, IGain)
    assert (state.gain, state.offset) == (camera._gain, 25.0)


@pytest.mark.asyncio
async def test_set_image_format() -> None:
    camera = AsiCamera(camera="test")
    await camera.set_image_format(ImageFormat.RGB24)
    assert camera._image_format == ImageFormat.RGB24


@pytest.mark.asyncio
async def test_set_image_format_invalid() -> None:
    camera = AsiCamera(camera="test")
    with pytest.raises(ValueError):
        await camera.set_image_format("bogus")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_reset_restores_gain_offset_image_format() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    camera._gain = 5.0
    camera._gain_offset = 10.0
    camera._image_format = ImageFormat.RGB24

    await camera.reset()

    assert camera._gain == 1.0
    assert camera._gain_offset == 50.0
    assert camera._image_format == ImageFormat.INT16


@pytest.mark.asyncio
async def test_full_reset_restores_cooling() -> None:
    camera = AsiCoolCamera(camera="test", setpoint=-15)
    camera.reset = AsyncMock()  # type: ignore[method-assign]
    camera.set_cooling = AsyncMock()  # type: ignore[method-assign]

    await camera.full_reset()

    camera.reset.assert_awaited_once()
    camera.set_cooling.assert_awaited_once_with(True, -15)


@pytest.mark.asyncio
async def test_run_blocking_runs_func_and_returns_true() -> None:
    ran: list[bool] = []

    def fast() -> None:
        ran.append(True)

    assert await AsiCamera._run_blocking(fast) is True
    assert ran == [True]


@pytest.mark.asyncio
async def test_run_blocking_times_out() -> None:
    done = threading.Event()

    def slow() -> None:
        done.wait()

    assert await AsiCamera._run_blocking(slow, timeout=0.01) is False
    done.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_run_blocking_or_raise_returns_value() -> None:
    camera = AsiCamera(camera="test")
    assert await camera._run_blocking_or_raise(lambda: 42) == 42


@pytest.mark.asyncio
async def test_run_blocking_or_raise_reraises() -> None:
    camera = AsiCamera(camera="test")

    def boom() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await camera._run_blocking_or_raise(boom)


def test_unknown_camera_has_no_modes() -> None:
    camera = AsiCamera(camera="test")
    assert camera._mode_options() == {}
    assert camera._current_modes() == {}


def test_preset_camera_defaults_to_low_noise() -> None:
    camera = AsiCamera(camera=ASI2600)
    assert (camera._gain, camera._gain_offset) == (100, 50)
    assert camera._mode_options() == {GAIN_GROUP: ["Low noise", "High capacity"]}
    assert camera._current_modes() == {GAIN_GROUP: "Low noise"}


def test_explicit_gain_and_offset_override_the_preset() -> None:
    camera = AsiCamera(camera=ASI2600, gain=0, offset=10)
    assert (camera._gain, camera._gain_offset) == (0, 10)
    assert camera._current_modes() == {}


@pytest.mark.asyncio
async def test_set_mode_sets_gain_offset_state_and_event() -> None:
    camera = AsiCamera(camera=ASI2600)
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    camera.comm.send_event = AsyncMock()  # type: ignore[method-assign]

    await camera.set_mode("High capacity")

    assert (camera._gain, camera._gain_offset) == (0, 50)
    assert _published(camera, IMode).modes == {GAIN_GROUP: "High capacity"}
    state = _published(camera, IGain)
    assert (state.gain, state.offset) == (0, 50)
    assert camera.comm.send_event.await_args is not None
    event = camera.comm.send_event.await_args.args[0]
    assert isinstance(event, ModeChangedEvent)
    assert (event.group, event.mode) == (GAIN_GROUP, "High capacity")


@pytest.mark.asyncio
async def test_manual_gain_leaves_the_mode_unset() -> None:
    camera = AsiCamera(camera=ASI2600)
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_gain(150)

    assert _published(camera, IMode).modes == {}


@pytest.mark.asyncio
async def test_set_mode_invalid() -> None:
    camera = AsiCamera(camera=ASI2600)
    with pytest.raises(exc.InvalidArgumentError):
        await camera.set_mode("Turbo")
    with pytest.raises(exc.InvalidArgumentError):
        await camera.set_mode("Low noise", group="Speed")
    with pytest.raises(exc.InvalidArgumentError):
        await AsiCamera(camera="test").set_mode("Low noise")


@pytest.mark.asyncio
async def test_reset_returns_to_the_default_mode() -> None:
    camera = AsiCamera(camera=ASI2600)
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    await camera.set_mode("High capacity")

    await camera.reset()

    assert camera._current_modes() == {GAIN_GROUP: "Low noise"}


def test_detector_gain_follows_the_measured_curve() -> None:
    preset = PRESETS[ASI2600]
    assert detector_gain(preset, {}, 0) == pytest.approx(0.783)
    # measured: 0.255 e-/ADU at gain 100, 0.081 at 200, 0.026 at 300
    assert detector_gain(preset, {}, 100) == pytest.approx(0.248, abs=0.01)
    assert detector_gain(preset, {}, 200) == pytest.approx(0.0783, abs=0.005)


def test_detector_gain_without_preset_uses_the_sdk_value_unscaled() -> None:
    assert detector_gain(None, {"ElecPerADU": 0.243}, 100) == 0.243
    assert detector_gain(None, {}, 100) is None
