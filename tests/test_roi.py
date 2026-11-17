"""Window and ROI handling: the module's window is in unbinned pixels, the SDK's ROI in binned pixels."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import zwoasi as asi  # type: ignore

from pyobs_asi import AsiCamera
from pyobs_asi.asicamera import binned_roi, unbinned_window


class FakeCamera:
    """Stands in for a zwoasi camera that returns a frame of the ROI size."""

    def __init__(self) -> None:
        self.roi_calls: list[tuple[Any, ...]] = []
        self.whbi = (0, 0, 1, asi.ASI_IMG_RAW16)

    def set_roi(self, *args: Any) -> None:
        self.roi_calls.append(args)
        self.whbi = (args[2], args[3], args[4], args[5])

    def set_control_value(self, control: int, value: int) -> None:
        pass

    def get_control_value(self, control: int) -> tuple[int, bool]:
        return 100, False  # temperature of 10 degC

    def start_exposure(self) -> None:
        pass

    def get_exposure_status(self) -> int:
        return asi.ASI_EXP_SUCCESS

    def get_data_after_exposure(self) -> bytes:
        return np.zeros(self.whbi[0] * self.whbi[1], dtype=np.uint16).tobytes()

    def get_roi_format(self) -> tuple[int, int, int, int]:
        return self.whbi


def test_binned_roi_without_binning_only_shrinks_to_the_sdk_grid() -> None:
    assert binned_roi((10, 20, 100, 51), 1) == (10, 20, 96, 50)
    assert unbinned_window(binned_roi((10, 20, 100, 51), 1), 1) == (10, 20, 96, 50)


def test_binned_roi_divides_start_and_size_by_the_binning() -> None:
    assert binned_roi((10, 20, 100, 50), 2) == (5, 10, 48, 24)
    # what the frame covers, in unbinned pixels
    assert unbinned_window((5, 10, 48, 24), 2) == (10, 20, 96, 48)


def test_binned_roi_keeps_the_minimum_size() -> None:
    assert binned_roi((0, 0, 4, 1), 2) == (0, 0, 8, 2)


def _camera(binning: int, window: tuple[int, int, int, int]) -> tuple[AsiCamera, FakeCamera]:
    module = AsiCamera(camera="test")
    fake = FakeCamera()
    module._camera = fake  # type: ignore[assignment]
    module._camera_info = {"PixelSize": 3.76, "MaxWidth": 6248, "MaxHeight": 4176}
    module._window = window
    module._binning = binning
    module.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    module._change_exposure_status = AsyncMock()  # type: ignore[method-assign]
    return module, fake


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "binning, window, roi, origin",
    [
        (1, (10, 20, 96, 50), (10, 20, 96, 50), (10, 20)),
        (2, (10, 20, 100, 50), (5, 10, 48, 24), (10, 20)),
        (3, (30, 12, 300, 60), (10, 4, 96, 20), (30, 12)),
    ],
)
async def test_expose_sets_the_binned_roi_and_the_unbinned_origin(
    binning: int, window: tuple[int, int, int, int], roi: tuple[int, int, int, int], origin: tuple[int, int]
) -> None:
    module, fake = _camera(binning, window)

    image = await module._expose(0.001, False, asyncio.Event())

    assert fake.roi_calls == [(*roi, binning, asi.ASI_IMG_RAW16)]
    assert image.data.shape == (roi[3], roi[2])
    assert (image.header["XORGSUBF"], image.header["YORGSUBF"]) == origin
    assert image.header["XBINNING"] == binning
