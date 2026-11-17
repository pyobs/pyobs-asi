"""Tests for the FITS header the GUI writes."""

from __future__ import annotations

from datetime import UTC, datetime

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


def test_header_has_settings_and_cooling() -> None:
    h = make_header(FakeCamera(), INFO, DATE, 300.0, (0, 0, 6248, 4176), 1, asi.ASI_IMG_RAW16)  # type: ignore[arg-type]
    assert h["DATE-OBS"] == "2026-11-17T10:00:00.500000"
    assert h["EXPTIME"] == 300.0
    assert h["GAIN"] == 100 and h["OFFSET"] == 50
    assert h["CCD-TEMP"] == -9.8 and h["SET-TEMP"] == -10.0
    assert h["COOLER"] is True and h["COOLPOW"] == 12
    assert h["INSTRUME"] == "ZWO ASI2600MM Pro" and h["IMAGETYP"] == "RAW16"


def test_header_binning_and_window_in_unbinned_pixels() -> None:
    h = make_header(FakeCamera(), INFO, DATE, 1.0, (10, 20, 100, 100), 2, asi.ASI_IMG_RAW8)  # type: ignore[arg-type]
    assert h["XBINNING"] == 2 and h["XORGSUBF"] == 20 and h["YORGSUBF"] == 40
    assert h["XPIXSZ"] == 7.52 and h["IMAGETYP"] == "RAW8"


def test_header_without_cooler_has_no_cooling_cards() -> None:
    h = make_header(FakeCamera(), {"Name": "x"}, DATE, 1.0, (0, 0, 8, 2), 1, asi.ASI_IMG_RAW16)  # type: ignore[arg-type]
    assert "SET-TEMP" not in h and "COOLPOW" not in h and "XPIXSZ" not in h
