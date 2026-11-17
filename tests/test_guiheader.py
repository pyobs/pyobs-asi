"""Tests for the FITS header the GUI writes."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import zwoasi as asi  # type: ignore

from pyobs_asi.gui import make_header


class FakeCamera:
    values = {
        asi.ASI_GAIN: 100,
        asi.ASI_OFFSET: 50,
        asi.ASI_TEMPERATURE: -98,
        asi.ASI_TARGET_TEMP: -10,
        asi.ASI_COOLER_ON: 1,
        asi.ASI_COOLER_POWER_PERC: 12,
    }

    def get_control_value(self, control: int) -> tuple[int, bool]:
        return self.values[control], False


INFO = {"Name": "ZWO ASI2600MM Pro", "PixelSize": 3.76, "IsCoolerCam": True}
DATE = datetime(2026, 11, 17, 10, 0, 0, 500000, tzinfo=UTC)
DATA = np.array([[490, 500], [510, 520]], dtype=np.uint16)


def test_header_has_the_module_keys() -> None:
    h = make_header(FakeCamera(), INFO, DATE, 300.0, (0, 0, 6248, 4176), 1, DATA)  # type: ignore[arg-type]
    assert h["DATE-OBS"] == "2026-11-17T10:00:00.500000"
    assert h["EXPTIME"] == 300.0
    assert h["GAIN"] == 100 and h["OFFSET"] == 50
    assert h["DET-GAIN"] == 0.783 * 10 ** (-100 / 200)
    assert h["DET-TEMP"] == -9.8 and h["DET-TSET"] == -10.0 and h["DET-COOL"] == 12
    assert h["INSTRUME"] == "ZWO ASI2600MM Pro"
    assert h["DET-PIXL"] == 0.00376
    assert (h["DATAMIN"], h["DATAMAX"], h["DATAMEAN"]) == (490.0, 520.0, 505.0)


def test_header_binning_and_window_in_unbinned_pixels() -> None:
    h = make_header(FakeCamera(), INFO, DATE, 1.0, (10, 20, 100, 100), 2, DATA)  # type: ignore[arg-type]
    assert h["XBINNING"] == h["DET-BIN1"] == 2 and h["XORGSUBF"] == 20 and h["YORGSUBF"] == 40


def test_header_without_cooler_has_no_cooling_cards() -> None:
    h = make_header(FakeCamera(), {"Name": "x"}, DATE, 1.0, (0, 0, 8, 2), 1, DATA)  # type: ignore[arg-type]
    assert "DET-TSET" not in h and "DET-COOL" not in h and "DET-PIXL" not in h
