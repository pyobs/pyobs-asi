"""Unit tests for the non-hardware logic in AsiCamera: constructor defaults, the
state-setting methods (window/binning/gain/offset/image format), and the
_run_blocking/_run_blocking_or_raise thread wrappers.

Hardware I/O (opening a camera, exposing, reading out) is out of scope here.
"""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest
from pyobs.interfaces import IBinning, IGain, IWindow
from pyobs.utils.enums import ImageFormat

from pyobs_asi import AsiCamera, AsiCoolCamera


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
    assert camera.comm.set_state.await_args is not None
    interface, state = camera.comm.set_state.await_args.args
    assert interface is IGain
    assert (state.gain, state.offset) == (5.0, camera._gain_offset)


@pytest.mark.asyncio
async def test_set_offset() -> None:
    camera = AsiCamera(camera="test")
    camera.comm.set_state = AsyncMock()  # type: ignore[method-assign]

    await camera.set_offset(25.0)

    assert camera._gain_offset == 25.0
    assert camera.comm.set_state.await_args is not None
    interface, state = camera.comm.set_state.await_args.args
    assert interface is IGain
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
