"""Per-camera gain presets, selectable as modes, and the detector gain in e-/ADU."""

from dataclasses import dataclass
from typing import Any

GAIN_GROUP = "Gain"


@dataclass(frozen=True)
class GainMode:
    """Gain and offset of a named mode."""

    gain: float
    offset: float


@dataclass(frozen=True)
class CameraPreset:
    """Modes and gain law of a known camera.

    Attributes:
        modes: Named gain/offset combinations.
        default_mode: Mode used at start-up and after a reset.
        e_per_adu_at_zero: Conversion gain in e-/ADU at gain 0.
    """

    modes: dict[str, GainMode]
    default_mode: str
    e_per_adu_at_zero: float

    def e_per_adu(self, gain: float) -> float:
        """Conversion gain in e-/ADU. A gain unit is 0.1 dB, i.e. a factor 10^(-gain/200)."""
        return self.e_per_adu_at_zero * 10.0 ** (-gain / 200.0)


# Keys are the camera names reported by the SDK.
PRESETS: dict[str, CameraPreset] = {
    # Measured with photon transfer at gains 0, 50, 100, 200 and 300 (see ASI2600MM_Pro.md): 0.783 e-/ADU at gain 0,
    # read noise 3.5 e- and 50.9 ke full scale there, 1.4 e- and 16.6 ke at gain 100 (the HCG mode).
    "ZWO ASI2600MM Pro": CameraPreset(
        modes={
            "Low noise": GainMode(gain=100, offset=50),
            "High capacity": GainMode(gain=0, offset=50),
        },
        default_mode="Low noise",
        e_per_adu_at_zero=0.783,
    ),
}


def detector_gain(preset: CameraPreset | None, camera_info: dict[str, Any], gain: float) -> float | None:
    """Conversion gain in e-/ADU for the DET-GAIN header, None if it is not known.

    For cameras without a preset this is the SDK value ElecPerADU as it is, because the gain it belongs to is not
    documented. It is not scaled with the gain setting.
    """
    if preset is not None:
        return preset.e_per_adu(gain)
    value = camera_info.get("ElecPerADU")
    return float(value) if value is not None else None
