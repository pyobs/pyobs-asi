"""Unit tests for the non-hardware logic in AsiVideo: frames(), the setters and the settings
restart. The ZWO SDK is replaced by a fake camera.
"""

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import numpy.typing as npt
import pytest
import zwoasi as asi  # type: ignore
from pyobs.utils.enums import ImageFormat

from pyobs_asi import AsiVideo


class FakeCamera:
    """Stands in for zwoasi.Camera: frames come from a list, calls are recorded."""

    def __init__(self, frames: list[npt.NDArray[Any]] | None = None) -> None:
        self.frames = [] if frames is None else frames
        self.calls: list[Any] = []
        self.controls: dict[int, int] = {}
        self.fail_roi = False

    def capture_video_frame(self, timeout: int) -> npt.NDArray[Any]:
        if self.frames:
            return self.frames.pop(0)
        threading.Event().wait(0.005)
        raise asi.ZWO_IOError("Timeout", 11)

    def start_video_capture(self) -> None:
        self.calls.append("start")

    def stop_video_capture(self) -> None:
        self.calls.append("stop")

    def set_roi(self, *args: Any) -> None:
        if self.fail_roi:
            raise ValueError("ROI and start position larger than binned sensor width")
        self.calls.append(("roi", *args))

    def set_control_value(self, control: int, value: int) -> None:
        # cameras round the exposure time to their own step size
        self.controls[control] = round(value / 7) * 7 if control == asi.ASI_EXPOSURE else value

    def get_control_value(self, control: int) -> list[Any]:
        return [self.controls[control], False]


def _video(fake: FakeCamera) -> AsiVideo:
    video = AsiVideo(camera="test", exposure_time=0.001)
    video._camera = fake  # type: ignore[assignment]
    video._window = (0, 0, 100, 50)
    comm = MagicMock()
    comm.set_state = AsyncMock()
    video._comm = comm
    return video


def _frame(value: int) -> npt.NDArray[Any]:
    return np.full((2, 2), value, dtype=np.uint16)


@pytest.mark.asyncio
async def test_frames_yields_in_order_and_stops_capture_on_close() -> None:
    fake = FakeCamera([_frame(i) for i in range(3)])
    video = _video(fake)

    iterator = video.frames()
    frames = [await anext(iterator) for _ in range(3)]
    await iterator.aclose()

    assert [int(f.data[0, 0]) for f in frames] == [0, 1, 2]
    assert all(f.generation == 0 and f.exposure_time == 0.001 and f.start is None for f in frames)
    assert fake.calls == ["start", "stop"]
    assert video._capturing is False


@pytest.mark.asyncio
async def test_frames_raises_sdk_errors_other_than_timeout() -> None:
    fake = FakeCamera()
    fake.capture_video_frame = MagicMock(side_effect=asi.ZWO_IOError("Camera removed", 5))  # type: ignore[method-assign]
    video = _video(fake)

    iterator = video.frames()
    with pytest.raises(asi.ZWO_IOError, match="Camera removed"):
        await anext(iterator)
    assert fake.calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_set_exposure_time_restarts_capture_and_bumps_generation() -> None:
    fake = FakeCamera()
    video = _video(fake)
    video._capturing = True

    await video.set_exposure_time(0.5)

    assert fake.calls == ["stop", ("roi", 0, 0, 96, 50, 1, asi.ASI_IMG_RAW16), "start"]
    assert video.generation == 1
    # the read-back value, not the requested one
    assert video._exposure_time == fake.controls[asi.ASI_EXPOSURE] / 1e6 != 0.5
    states = {type(c.args[1]).__name__: c.args[1] for c in video.comm.set_state.await_args_list}  # type: ignore[attr-defined]
    assert states["ExposureTimeState"].exposure_time == video._exposure_time


@pytest.mark.asyncio
async def test_set_binning_sets_binned_roi_without_restart() -> None:
    fake = FakeCamera()
    video = _video(fake)
    video._window = (10, 20, 100, 50)

    await video.set_binning(2, 2)

    # ROI in binned pixels, width a multiple of 8, height a multiple of 2
    assert fake.calls == [("roi", 5, 10, 48, 24, 2, asi.ASI_IMG_RAW16)]


@pytest.mark.asyncio
async def test_set_binning_rejects_non_square() -> None:
    video = _video(FakeCamera())
    with pytest.raises(ValueError):
        await video.set_binning(1, 2)


@pytest.mark.asyncio
async def test_set_image_format_rejects_rgb() -> None:
    video = _video(FakeCamera())
    with pytest.raises(ValueError):
        await video.set_image_format(ImageFormat.RGB24)


@pytest.mark.asyncio
async def test_rejected_settings_are_restored() -> None:
    fake = FakeCamera()
    video = _video(fake)
    await video.set_window(0, 0, 100, 50)

    fake.fail_roi = True
    with pytest.raises(ValueError):
        await video.set_window(0, 0, 10000, 50)
    assert video._window == (0, 0, 100, 50)


@pytest.mark.asyncio
async def test_setters_before_open_only_store() -> None:
    video = _video(FakeCamera())
    video._camera = None

    await video.set_gain(5.0)

    assert video._gain == 5.0
    assert video.generation == 0


@pytest.mark.asyncio
async def test_frames_after_settings_change_carry_new_generation() -> None:
    fake = FakeCamera([_frame(0)])
    video = _video(fake)
    iterator = video.frames()

    first = await anext(iterator)
    await video.set_exposure_time(0.5)
    fake.frames.append(_frame(1))
    second = await anext(iterator)
    await iterator.aclose()

    assert (first.generation, first.exposure_time) == (0, 0.001)
    assert (second.generation, second.exposure_time) == (1, fake.controls[asi.ASI_EXPOSURE] / 1e6)
    assert fake.calls[:2] == ["start", "stop"]
    assert fake.calls[-2:] == ["start", "stop"]


@pytest.mark.asyncio
async def test_run_blocking_late_completion_does_not_raise() -> None:
    # a func finishing after the timeout must not call set_result on the cancelled future
    done = threading.Event()
    errors: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: errors.append(ctx))

    def slow() -> None:
        done.wait()

    assert await AsiVideo._run_blocking(slow, timeout=0.01) is False
    done.set()
    await asyncio.sleep(0.05)

    assert errors == []
