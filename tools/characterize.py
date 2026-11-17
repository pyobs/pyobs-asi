"""Characterize a cooled ASI camera in the dark.

Run with the lens capped and without any other program using the camera, e.g.::

    uv run python tools/characterize.py bias --sdk sdk/x64/libASICamera2.so
    uv run python tools/characterize.py gain
    uv run python tools/characterize.py temperature
    uv run python tools/characterize.py flat   # lens uncapped, pointed at a steady, defocused light source
    uv run python tools/characterize.py short  # same light source
    uv run python tools/characterize.py readout
    uv run python tools/characterize.py pixels --target-temp -5  # lens capped
    uv run python tools/characterize.py report

Steps (all with high speed mode off, on a central 2048x2048 window):

* ``bias``: bias level, read noise and clipping vs. gain and offset.
* ``gain``: gain in ADU/e- from the shot noise of the dark current (photon transfer without light). Needs a warm
  sensor, so it turns the cooler off and waits until the temperature is stable.
* ``temperature``: read noise, dark current and hot pixels vs. sensor temperature. The cooler target is moved in steps
  of at most 2 degC/min and the camera is warmed up again afterwards.
* ``flat``: photon transfer curve from pairs of frames of a steady, roughly uniform light source (e.g. a wall): gain in
  e-/ADU, read noise in e-, saturation and linearity. The bias level comes from the fit of level vs. exposure time.
* ``short``: signal vs. exposure times from the minimum up to 100 ms with a steady light source, to find an offset in
  the exposure time or a nonlinearity for short exposures.
* ``readout``: time per frame for the full frame vs. binning and bit depth, with high speed mode off and on, and the
  read noise in both modes.
* ``pixels``: temporal noise of every pixel from a stack of full-frame bias frames: the share of pixels with
  excess noise (random telegraph signal, "salt and pepper"), fixed offsets, and how many of the noisy ones flip between
  two levels. Use ``--target-temp`` to cool first, or ``--no-cooling`` for the warm sensor.
* ``report``: plots (in ``images/``) and a CSV file for every JSON lines file, from the measured data. Needs no camera.

Results are appended to JSON lines files in ``characterization/<model>/<serial number>/``, so a run can be repeated or
resumed. See ``characterization/README.md``.
"""

import argparse
import csv
import ctypes
import json
import math
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zwoasi as asi  # type: ignore
from astropy.io import fits

ROI = 2048
CUTOUT = 512
RAMP_STEP_C = 1.0
RAMP_STEP_S = 30.0  # together: 2 degC/min
MAX_POWER = 90
STABLE_RANGE_C = 0.2
STABLE_TIME_S = 30.0
POLL_S = 5.0  # temperature readings while waiting for a stable sensor
LIMIT_TIME_S = 120.0  # the cooler must be above MAX_POWER for this long before the series stops
TARGET_JITTER_C = 1.2  # the sensor reading wobbles by about +-0.5 degC when the cooler idles near ambient

GAINS = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700]
OFFSETS = [0, 10, 50, 100, 240]
GAIN_TEST_GAINS = [0, 100, 300]
GAIN_TEST_EXPOSURES = [30.0, 100.0, 300.0]
TEMPERATURES = [20, 10, 0, -10, -20]
DARK_EXPOSURES = [60.0, 300.0]
DEFAULT_GAIN = 100
FULL_SCALE = 65535
FLAT_GAINS = [0, 50, 100, 200, 300]
FLAT_POINTS = 12
FLAT_RANGE = (0.03, 0.85)  # fractions of full scale covered by the exposure ladder
FLAT_MAX_EXPOSURE_S = 20.0
LINEAR_MAX_FRACTION = 0.6  # bias and rate are fitted below this level
FIT_MAX_FRACTION = 0.75  # the photon transfer fit uses signals below this fraction of full scale
DEFAULT_OFFSET = 50
SHORT_EXPOSURES_S = [0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SHORT_REFERENCE_S = 0.01  # repeated between the points to follow the light source
SHORT_FRAMES = 5
SHORT_FIT_MIN_S = 0.001  # the exposure offset is fitted from exposures from here on
READOUT_FRAMES = 5
PIXEL_FRAMES = 60
PIXEL_SAMPLE = 100_000  # pixels whose full time series is kept to look for two-level (telegraph) behaviour
PIXEL_NOISY_FACTOR = 5.0  # a pixel is noisy if its temporal std is this many times the median


class CoolerLimit(Exception):
    """The cooler cannot reach the requested temperature."""


class Camera:
    def __init__(self, sdk: str) -> None:
        asi.init(sdk)
        if asi.get_num_cameras() == 0:
            raise SystemExit("No camera found.")
        self.cam = asi.Camera(0)
        self.info: dict[str, Any] = self.cam.get_camera_property()
        self.controls: dict[str, Any] = self.cam.get_controls()
        self.serial = serial_number(self.cam)
        self.cam.disable_dark_subtract()
        self.cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)
        self.min_exposure_us = int(self.controls["Exposure"]["MinValue"])
        self.left = (self.info["MaxWidth"] - ROI) // 2 // 8 * 8
        self.top = (self.info["MaxHeight"] - ROI) // 2 // 8 * 8
        self.sdk = sdk
        # the SDK reports 0.0 until the first temperature update arrives
        for _ in range(100):
            if self.temperature() != 0.0:
                break
            time.sleep(0.1)

    def grab(self, exposure_s: float, gain: int, offset: int) -> np.ndarray:
        c = self.cam
        c.set_control_value(asi.ASI_GAIN, gain)
        c.set_control_value(asi.ASI_OFFSET, offset)
        c.set_control_value(asi.ASI_EXPOSURE, max(int(exposure_s * 1e6), self.min_exposure_us))
        c.set_roi(self.left, self.top, ROI, ROI, bins=1, image_type=asi.ASI_IMG_RAW16)
        c.start_exposure()
        time.sleep(min(exposure_s, 1.0))
        while c.get_exposure_status() == asi.ASI_EXP_WORKING:
            time.sleep(0.05 if exposure_s < 5 else 0.5)
        if c.get_exposure_status() != asi.ASI_EXP_SUCCESS:
            raise RuntimeError("Exposure failed.")
        data = np.frombuffer(c.get_data_after_exposure(), dtype=np.uint16).reshape(ROI, ROI)
        return data

    def temperature(self) -> float:
        return float(self.cam.get_control_value(asi.ASI_TEMPERATURE)[0]) / 10.0

    def power(self) -> int:
        return int(self.cam.get_control_value(asi.ASI_COOLER_POWER_PERC)[0])

    def cooler_on(self) -> bool:
        return bool(self.cam.get_control_value(asi.ASI_COOLER_ON)[0])

    def set_cooler(self, on: bool, target: int | None = None) -> None:
        if target is not None:
            self.cam.set_control_value(asi.ASI_TARGET_TEMP, int(target))
        self.cam.set_control_value(asi.ASI_COOLER_ON, int(on))


class Output:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        (folder / "images").mkdir(exist_ok=True)

    def write(self, name: str, record: dict[str, Any]) -> None:
        with open(self.folder / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self, name: str) -> list[dict[str, Any]]:
        path = self.folder / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def save_cutout(self, name: str, data: np.ndarray, **header: Any) -> None:
        y, x = (data.shape[0] - CUTOUT) // 2, (data.shape[1] - CUTOUT) // 2
        hdu = fits.PrimaryHDU(data[y : y + CUTOUT, x : x + CUTOUT])
        for key, value in header.items():
            hdu.header[key[:8].upper()] = value
        hdu.writeto(self.folder / "frames" / f"{name}.fits", overwrite=True)


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Noise from the difference of two frames (removes fixed pattern), mean level, clipping."""
    d = a.astype(np.float64) - b.astype(np.float64)
    mad = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    return {
        "mean": float((a.mean() + b.mean()) / 2),
        "median": float((np.median(a) + np.median(b)) / 2),
        "noise_std_adu": float(d.std() / math.sqrt(2)),
        "noise_mad_adu": mad / math.sqrt(2),
        "clipped_frac": float(((a == 0).mean() + (b == 0).mean()) / 2),
    }


def bias_pair(cam: Camera, gain: int, offset: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    cam.grab(0.0, gain, offset)  # first frame after a setting change is discarded
    a = cam.grab(0.0, gain, offset)
    b = cam.grab(0.0, gain, offset)
    return a, b, pair_stats(a, b)


def wait_stable(cam: Camera, out: Output, target: int | None, label: str) -> None:
    """Wait until the sensor temperature stops changing (and has reached the target)."""
    history: deque[tuple[float, float]] = deque()
    high_since: float | None = None
    while True:
        now, temp = time.time(), cam.temperature()
        out.write("templog", {"time": now, "label": label, "temp": temp, "power": cam.power(), "target": target})
        history.append((now, temp))
        # keep slightly more than the required time, samples come every POLL_S
        while history and now - history[0][0] > STABLE_TIME_S + 5:
            history.popleft()
        temps = [t for _, t in history]
        full = now - history[0][0] >= STABLE_TIME_S * 0.9
        at_target = target is None or abs(float(np.mean(temps)) - target) <= 0.5
        # without a target (warm sensor) only a slow drift is needed, the dark signal is measured per frame pair
        tolerance = TARGET_JITTER_C if target is not None else 2.5 * STABLE_RANGE_C
        if full and at_target and max(temps) - min(temps) <= tolerance:
            return
        if target is None or cam.power() <= MAX_POWER or temp - target <= 1.0:
            high_since = None
        elif high_since is None:
            high_since = now
        if high_since is not None and now - high_since >= LIMIT_TIME_S:
            raise CoolerLimit(f"cooler at {cam.power()}% and sensor at {temp:.1f} degC, target {target} degC")
        print(f"  {label}: T={temp:.1f} degC power={cam.power()}% target={target}", flush=True)
        time.sleep(POLL_S)


def ramp_to(cam: Camera, out: Output, target: int) -> None:
    """Move the cooler target to the new value in steps, at most 2 degC/min."""
    current = (
        round(cam.temperature()) if not cam.cooler_on() else int(cam.cam.get_control_value(asi.ASI_TARGET_TEMP)[0])
    )
    cam.set_cooler(True, current)
    while current != target:
        current += int(math.copysign(RAMP_STEP_C, target - current))
        cam.set_cooler(True, current)
        time.sleep(RAMP_STEP_S)
        out.write(
            "templog",
            {"time": time.time(), "label": "ramp", "temp": cam.temperature(), "power": cam.power(), "target": current},
        )


def warm_up(cam: Camera, out: Output, ambient: int) -> None:
    print(f"Warming up to {ambient} degC at 2 degC/min and switching the cooler off.", flush=True)
    if cam.cooler_on():
        ramp_to(cam, out, ambient)
    cam.set_cooler(False)


def run_bias(cam: Camera, out: Output, gains: list[int], offsets: list[int]) -> None:
    for gain in gains:
        for offset in offsets:
            a, _, stats = bias_pair(cam, gain, offset)
            record = {"gain": gain, "offset": offset, "temp": cam.temperature(), "time": time.time(), **stats}
            out.write("bias", record)
            print(
                f"gain={gain:3d} offset={offset:3d} mean={stats['mean']:8.1f} noise={stats['noise_std_adu']:5.2f} ADU "
                f"clipped={stats['clipped_frac']:.4f}",
                flush=True,
            )
            if offset == DEFAULT_OFFSET:
                out.save_cutout(f"bias_g{gain}_o{offset}", a, gain=gain, offset=offset, temp=cam.temperature())


def run_gain(
    cam: Camera, out: Output, gains: list[int], exposures: list[float], offset: int, target_temp: int | None
) -> None:
    ambient = round(cam.temperature())
    try:
        if target_temp is None:
            if cam.cooler_on():
                print("Switching the cooler off and waiting for a stable sensor temperature.", flush=True)
                cam.set_cooler(False)
            wait_stable(cam, out, None, "warm")
        else:
            print(f"Cooling to {target_temp} degC ...", flush=True)
            ramp_to(cam, out, target_temp)
            wait_stable(cam, out, target_temp, f"stabilize {target_temp}")
        _gain_frames(cam, out, gains, exposures, offset)
    finally:
        if target_temp is not None:
            warm_up(cam, out, ambient)


def _gain_frames(cam: Camera, out: Output, gains: list[int], exposures: list[float], offset: int) -> None:
    for gain in gains:
        _, _, bias = bias_pair(cam, gain, offset)
        for exposure in exposures:
            cam.grab(exposure, gain, offset)  # discard the first frame after a setting change
            a = cam.grab(exposure, gain, offset)
            b = cam.grab(exposure, gain, offset)
            stats = pair_stats(a, b)
            record = {
                "gain": gain,
                "offset": offset,
                "exposure": exposure,
                "temp": cam.temperature(),
                "power": cam.power(),
                "time": time.time(),
                "bias_mean": bias["mean"],
                "signal_adu": stats["mean"] - bias["mean"],
                "variance_adu2": stats["noise_std_adu"] ** 2,
                "variance_mad_adu2": stats["noise_mad_adu"] ** 2,
            }
            out.write("gain", record)
            out.save_cutout(f"gain_g{gain}_t{int(exposure)}", a, gain=gain, offset=offset, exptime=exposure)
            print(
                f"gain={gain:3d} t={exposure:5.0f}s signal={record['signal_adu']:8.2f} ADU "
                f"variance={record['variance_adu2']:8.2f} ADU^2 T={record['temp']:.1f}",
                flush=True,
            )


def crop(data: np.ndarray) -> np.ndarray:
    y, x = (data.shape[0] - CUTOUT) // 2, (data.shape[1] - CUTOUT) // 2
    return data[y : y + CUTOUT, x : x + CUTOUT]


def flat_ladder(cam: Camera, gain: int, offset: int) -> list[float]:
    """Exposure times (s) that cover FLAT_RANGE of the full scale, from a probe of the light level."""
    guess = offset * 10.0  # bias level estimate, only used to set the ladder
    min_s = cam.min_exposure_us * 1e-6
    t, rate = 0.01, 0.0
    for _ in range(5):
        cam.grab(t, gain, offset)  # discard
        level = float(np.median(crop(cam.grab(t, gain, offset))))
        rate = (level - guess) / t
        if rate <= 0:
            raise SystemExit("No light on the sensor, check the lens cap and the light source.")
        if 0.2 * FULL_SCALE < level < 0.6 * FULL_SCALE:
            break
        t = float(np.clip(0.4 * (FULL_SCALE - guess) / rate, min_s, FLAT_MAX_EXPOSURE_S))
    levels = np.geomspace(*FLAT_RANGE, FLAT_POINTS) * (FULL_SCALE - guess)
    exposures = [round(float(level / rate), 6) for level in levels]
    usable = [e for e in exposures if min_s <= e <= FLAT_MAX_EXPOSURE_S]
    if len(usable) < FLAT_POINTS:
        print(f"  gain {gain}: only {len(usable)} of {FLAT_POINTS} exposures are within the camera limits.", flush=True)
    return usable


def run_flat(cam: Camera, out: Output, gains: list[int], offset: int) -> None:
    for gain in gains:
        ladder = flat_ladder(cam, gain, offset)
        if len(ladder) < 4:
            print(f"gain={gain:3d}: too few usable exposures, skipping.", flush=True)
            continue
        print(f"gain={gain:3d}: {len(ladder)} exposures from {ladder[0]:.6f} to {ladder[-1]:.4f} s", flush=True)
        pairs = []
        for exposure in ladder:
            cam.grab(exposure, gain, offset)  # discard the first frame after a setting change
            a, b = crop(cam.grab(exposure, gain, offset)), crop(cam.grab(exposure, gain, offset))
            pairs.append((exposure, a, b, cam.temperature(), time.time()))

        # level = bias + rate * exposure, fitted in the linear regime; the intercept is the bias without a cap
        exposures = np.array([p[0] for p in pairs])
        means = np.array([(p[1].mean() + p[2].mean()) / 2 for p in pairs])
        low = means < LINEAR_MAX_FRACTION * FULL_SCALE
        if low.sum() < 3:
            low = np.ones_like(low)
        rate, bias = np.polyfit(exposures[low], means[low], 1)

        for (exposure, a, b, temp, now), mean in zip(pairs, means, strict=True):
            ratio = (a.mean() - bias) / (b.mean() - bias)  # lamp drift between the two frames
            stats = pair_stats(a, (b.astype(np.float64) - bias) * ratio + bias)
            record = {
                "gain": gain,
                "offset": offset,
                "exposure": exposure,
                "temp": temp,
                "time": now,
                "bias_fit": float(bias),
                "rate_adu_s": float(rate),
                "signal_adu": float(mean - bias),
                "variance_adu2": stats["noise_std_adu"] ** 2,
                "variance_mad_adu2": stats["noise_mad_adu"] ** 2,
                "drift_pct": float((ratio - 1) * 100),
                "sat_frac": float(((a >= FULL_SCALE).mean() + (b >= FULL_SCALE).mean()) / 2),
                "max_adu": int(max(a.max(), b.max())),
                "nonlin_pct": float((mean - (bias + rate * exposure)) / (rate * exposure) * 100),
            }
            out.write("flat", record)
            print(
                f"  t={exposure:9.6f}s signal={record['signal_adu']:8.1f} ADU variance={record['variance_adu2']:9.1f} "
                f"drift={record['drift_pct']:+.2f}% nonlin={record['nonlin_pct']:+.2f}%",
                flush=True,
            )
        out.save_cutout(f"flat_g{gain}_low", pairs[0][1], gain=gain, offset=offset, exptime=pairs[0][0])
        out.save_cutout(f"flat_g{gain}_high", pairs[-1][1], gain=gain, offset=offset, exptime=pairs[-1][0])


def run_temperature(cam: Camera, out: Output, temperatures: list[int], gain: int, offset: int) -> None:
    ambient = round(cam.temperature())
    try:
        for target in temperatures:
            print(f"Cooling to {target} degC ...", flush=True)
            ramp_to(cam, out, target)
            try:
                wait_stable(cam, out, target, f"stabilize {target}")
            except CoolerLimit as e:
                print(f"Stopping the series: {e}", flush=True)
                out.write("temperature", {"time": time.time(), "target": target, "limit": str(e)})
                break
            _, _, bias = bias_pair(cam, gain, offset)
            for exposure in DARK_EXPOSURES:
                cam.grab(exposure, gain, offset)  # discard
                da = cam.grab(exposure, gain, offset)
                db = cam.grab(exposure, gain, offset)
                dark = pair_stats(da, db)
                signal_mean = dark["mean"] - bias["mean"]
                signal_median = dark["median"] - bias["median"]
                hot = float((da > np.median(da) + 20 * bias["noise_mad_adu"]).mean())
                record = {
                    "target": target,
                    "temp": cam.temperature(),
                    "power": cam.power(),
                    "time": time.time(),
                    "gain": gain,
                    "offset": offset,
                    "exposure": exposure,
                    "bias_noise_adu": bias["noise_std_adu"],
                    "bias_mean": bias["mean"],
                    "dark_noise_adu": dark["noise_std_adu"],
                    "dark_adu_per_s_mean": signal_mean / exposure,
                    "dark_adu_per_s_median": signal_median / exposure,
                    "hot_frac": hot,
                }
                out.write("temperature", record)
                out.save_cutout(f"dark_T{target}_t{int(exposure)}", da, temp=record["temp"], exptime=exposure)
                print(
                    f"T={record['temp']:.1f} t={exposure:.0f}s bias noise={bias['noise_std_adu']:.2f} ADU "
                    f"dark={record['dark_adu_per_s_mean']:.5f} ADU/s (mean) hot={hot:.5f}",
                    flush=True,
                )
    finally:
        warm_up(cam, out, ambient)


def run_short(cam: Camera, out: Output, gain: int, offset: int, reference_s: float = SHORT_REFERENCE_S) -> None:
    """Mean level of the central window vs. exposure time for short exposures, with a reference exposure between."""
    min_s = cam.min_exposure_us * 1e-6
    # bias level from frames at the minimum exposure, the signal level is relative to that
    cam.grab(min_s, gain, offset)
    bias = float(np.mean([crop(cam.grab(min_s, gain, offset)).mean() for _ in range(SHORT_FRAMES)]))
    cam.grab(reference_s, gain, offset)
    ref0 = float(crop(cam.grab(reference_s, gain, offset)).mean())
    if ref0 - bias <= 20:
        raise SystemExit("Too little light on the sensor, check the lens cap and the light source.")
    rate = (ref0 - bias) / reference_s
    for exposure in SHORT_EXPOSURES_S:
        exposure = max(exposure, min_s)
        if bias + rate * exposure > 0.85 * FULL_SCALE:
            print(f"  t={exposure:.6f}s skipped, would saturate", flush=True)
            continue
        cam.grab(exposure, gain, offset)  # discard the first frame after a setting change
        levels = [float(crop(cam.grab(exposure, gain, offset)).mean()) for _ in range(SHORT_FRAMES)]
        cam.grab(reference_s, gain, offset)
        ref = float(crop(cam.grab(reference_s, gain, offset)).mean())
        record = {
            "gain": gain,
            "offset": offset,
            "exposure": exposure,
            "time": time.time(),
            "temp": cam.temperature(),
            "level": float(np.mean(levels)),
            "level_std": float(np.std(levels)),
            "level_min_exposure": bias,
            "ref_level": ref,
        }
        out.write("short", record)
        print(
            f"  t={exposure:9.6f}s level={record['level']:9.2f} +- {record['level_std']:.2f} ADU "
            f"reference={ref - bias:8.1f} ADU",
            flush=True,
        )


def _timed_frames(cam: Camera, width: int, height: int, bins: int, image_type: int, gain: int, offset: int) -> list:
    c = cam.cam
    c.set_control_value(asi.ASI_GAIN, gain)
    c.set_control_value(asi.ASI_OFFSET, offset)
    c.set_control_value(asi.ASI_EXPOSURE, cam.min_exposure_us)
    c.set_roi(0, 0, width, height, bins=bins, image_type=image_type)
    frames = []
    for i in range(READOUT_FRAMES + 1):  # the first frame after a setting change is discarded
        t0 = time.time()
        c.start_exposure()
        while c.get_exposure_status() == asi.ASI_EXP_WORKING:
            time.sleep(0.001)
        t1 = time.time()
        if c.get_exposure_status() != asi.ASI_EXP_SUCCESS:
            raise RuntimeError("Exposure failed.")
        data = c.get_data_after_exposure()
        t2 = time.time()
        if i > 0:
            frames.append((t1 - t0, t2 - t1, data))
    return frames


def run_readout(cam: Camera, out: Output, gain: int, offset: int) -> None:
    """Time per frame of the full frame vs. binning and bit depth, and the read noise, in both speed modes."""
    info = cam.info
    modes = [("RAW16", asi.ASI_IMG_RAW16, np.uint16), ("RAW8", asi.ASI_IMG_RAW8, np.uint8)]
    for speed in (0, 1):
        cam.cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, speed)
        for bins in info["SupportedBins"]:
            width = info["MaxWidth"] // bins // 8 * 8
            height = info["MaxHeight"] // bins // 2 * 2
            for name, image_type, dtype in modes:
                frames = _timed_frames(cam, width, height, bins, image_type, gain, offset)
                exposure_s = float(np.median([f[0] for f in frames]))
                download_s = float(np.median([f[1] for f in frames]))
                a = np.frombuffer(frames[0][2], dtype=dtype).reshape(height, width)
                b = np.frombuffer(frames[1][2], dtype=dtype).reshape(height, width)
                noise = pair_stats(crop(a), crop(b)) if min(a.shape) >= CUTOUT else None
                record = {
                    "time": time.time(),
                    "high_speed": speed,
                    "bins": bins,
                    "type": name,
                    "width": width,
                    "height": height,
                    "gain": gain,
                    "offset": offset,
                    "exposure_s": exposure_s,
                    "download_s": download_s,
                    "frame_s": exposure_s + download_s,
                    "noise_adu": None if noise is None else noise["noise_std_adu"],
                    "bias_mean": None if noise is None else noise["mean"],
                }
                out.write("readout", record)
                print(
                    f"  high_speed={speed} bin={bins} {name} {width}x{height}: exposure+readout {exposure_s:.3f} s, "
                    f"download {download_s:.3f} s, noise={record['noise_adu']}",
                    flush=True,
                )
    cam.cam.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)


def pixel_stats(
    sample: np.ndarray, s: np.ndarray, s2: np.ndarray, lo: np.ndarray, hi: np.ndarray, first: np.ndarray, n: int
) -> dict[str, Any]:
    """Statistics of the temporal noise of every pixel, from sums over ``n`` frames.

    ``sample`` has the time series (frames x pixels) of a random subset of pixels, ``first`` is the first frame.
    """
    mean = s / n
    std = np.sqrt(np.maximum((s2 - n * mean**2) / (n - 1), 0.0))
    level, med_std = float(np.median(mean)), float(np.median(std))
    record: dict[str, Any] = {
        "frames": n,
        "level": level,
        "std_median": med_std,
        "std_mean": float(std.mean()),
        "std_rms": float(np.sqrt((std**2).mean())),
        "std_percentiles": {str(q): float(v) for q, v in zip((1, 50, 99, 99.9), np.percentile(std, (1, 50, 99, 99.9)))},
        "noisy_frac": {str(k): float((std > k * med_std).mean()) for k in (1.25, 1.5, 2, 3, 5)},
        "offset_high": {str(a): int((mean > level + a).sum()) for a in (10, 20, 40)},
        "offset_low": {str(a): int((mean < level - a).sum()) for a in (10, 20, 40)},
    }
    # tails of a single frame, with a robust sigma (a Gaussian would give a few pixels at 5 sigma)
    sigma = 1.4826 * float(np.median(np.abs(first - np.median(first))))
    record["single_frame_tails"] = {
        str(k): {
            "high": int((first > np.median(first) + k * sigma).sum()),
            "low": int((first < np.median(first) - k * sigma).sum()),
        }
        for k in (5, 8, 20)
    }
    # noisy pixels of the sample: do they sit on two levels?
    sample_std = sample.std(axis=0, ddof=1)
    noisy = sample[:, sample_std > PIXEL_NOISY_FACTOR * med_std].astype(np.float64)
    two_level = 0
    for series in noisy.T:
        split = (series.min() + series.max()) / 2
        up, down = series[series > split], series[series <= split]
        share = min(len(up), len(down)) / len(series)
        if share < 0.1 or len(up) < 2 or len(down) < 2:
            continue
        pooled = math.sqrt((up.var(ddof=1) * (len(up) - 1) + down.var(ddof=1) * (len(down) - 1)) / (len(series) - 2))
        if pooled > 0 and (up.mean() - down.mean()) / pooled > 3:
            two_level += 1
    record["sample_noisy"] = int(noisy.shape[1])
    record["sample_noisy_two_level"] = two_level
    record["range_over_45adu"] = int(((hi.astype(np.int64) - lo) > 45).sum())
    return record


def run_pixels(
    cam: Camera, out: Output, gain: int, offset: int, target_temp: int | None, frames: int = PIXEL_FRAMES
) -> None:
    ambient = round(cam.temperature())
    try:
        if target_temp is None:
            if cam.cooler_on():
                print("Switching the cooler off and waiting for a stable sensor temperature.", flush=True)
                cam.set_cooler(False)
            wait_stable(cam, out, None, "warm")
        else:
            print(f"Cooling to {target_temp} degC ...", flush=True)
            ramp_to(cam, out, target_temp)
            wait_stable(cam, out, target_temp, f"stabilize {target_temp}")

        c = cam.cam
        width, height = cam.info["MaxWidth"], cam.info["MaxHeight"]
        c.set_control_value(asi.ASI_GAIN, gain)
        c.set_control_value(asi.ASI_OFFSET, offset)
        c.set_control_value(asi.ASI_EXPOSURE, cam.min_exposure_us)
        c.set_roi(0, 0, width, height, bins=1, image_type=asi.ASI_IMG_RAW16)
        rng = np.random.default_rng(1)
        index = rng.choice(width * height, PIXEL_SAMPLE, replace=False)
        s, s2 = np.zeros((height, width)), np.zeros((height, width))
        lo, hi = np.full((height, width), 65535, np.uint16), np.zeros((height, width), np.uint16)
        sample, first = [], None
        temp0 = cam.temperature()
        for i in range(frames + 1):  # the first frame after a setting change is discarded
            c.start_exposure()
            while c.get_exposure_status() == asi.ASI_EXP_WORKING:
                time.sleep(0.001)
            if c.get_exposure_status() != asi.ASI_EXP_SUCCESS:
                raise RuntimeError("Exposure failed.")
            data = np.frombuffer(c.get_data_after_exposure(), dtype=np.uint16).reshape(height, width)
            if i == 0:
                continue
            x = data.astype(np.float64)
            s += x
            s2 += x * x
            lo, hi = np.minimum(lo, data), np.maximum(hi, data)
            sample.append(data.reshape(-1)[index])
            if first is None:
                first = x
        assert first is not None
        record: dict[str, Any] = {
            "time": time.time(),
            "target": target_temp,
            "temp": cam.temperature(),
            "temp_start": temp0,
            "power": cam.power(),
            "gain": gain,
            "offset": offset,
            **pixel_stats(np.array(sample), s, s2, lo, hi, first, frames),
        }
        out.write("pixels", record)
        print(
            f"T={record['temp']:.1f} degC: median pixel noise {record['std_median']:.2f} ADU, "
            f"rms {record['std_rms']:.2f}, "
            f"noisy (>{PIXEL_NOISY_FACTOR:.0f}x): {record['noisy_frac']['5']:.2e}, two-level in sample: "
            f"{record['sample_noisy_two_level']} of {record['sample_noisy']}",
            flush=True,
        )
    finally:
        if target_temp is not None:
            warm_up(cam, out, ambient)


def fit_gain(records: list[dict[str, Any]], gain: int, weighted: bool = False) -> dict[str, float] | None:
    """Fit variance = read_noise^2 + g * signal. The slope g is ADU per electron.

    The variance of a variance estimate grows with the variance itself, so ``weighted`` uses weights of 1/variance.
    Without it a wide signal range makes the intercept (read noise) meaningless.
    """
    rows = [r for r in records if r["gain"] == gain]
    if len(rows) < 3:
        return None
    s = np.array([r["signal_adu"] for r in rows])
    v = np.array([r["variance_adu2"] for r in rows])
    w = 1.0 / v if weighted else np.ones_like(v)
    slope, intercept = np.polyfit(s, v, 1, w=w)
    residual = (v - (slope * s + intercept)) * w
    s_mean = np.sum(w**2 * s) / np.sum(w**2)
    err = float(np.sqrt(np.sum(residual**2) / max(len(s) - 2, 1) / np.sum(w**2 * (s - s_mean) ** 2)))
    return {
        "adu_per_e": float(slope),
        "e_per_adu": float(1.0 / slope) if slope > 0 else float("nan"),
        "slope_err": err,
        "read_noise_adu": float(math.sqrt(max(intercept, 0.0))),
        "signal_max_adu": float(s.max()),
    }


def export_csv(folder: Path) -> list[Path]:
    """Writes every JSON lines file in the folder as a CSV file of the same name.

    Nested values are flattened into columns named ``<key>_<subkey>``, lists are joined with spaces, and a ``utc``
    column is added in front if the records have a ``time``.
    """

    def flat(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, dict):
                out.update(flat(value, f"{prefix}{key}_"))
            elif isinstance(value, list):
                out[f"{prefix}{key}"] = " ".join(str(v) for v in value)
            else:
                out[f"{prefix}{key}"] = value
        return out

    written = []
    for path in sorted(folder.glob("*.jsonl")):
        rows = [flat(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        keys: list[str] = []
        for row in rows:
            if "time" in row:
                row["utc"] = datetime.fromtimestamp(row["time"], UTC).strftime("%Y-%m-%dT%H:%M:%S")
            keys += [k for k in row if k not in keys]
        if "utc" in keys:
            keys.remove("utc")
            keys.insert(0, "utc")
        csv_path = path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        written.append(csv_path)
    return written


def make_report(out: Output) -> None:
    """Draws the plots into ``images/`` and writes the CSV files, from the measured data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = out.folder / "images"
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        fig.savefig(images / name, dpi=110)
        plt.close(fig)
        written.append(images / name)

    bias, gain, temp = out.read("bias"), out.read("gain"), out.read("temperature")
    temp = [r for r in temp if "bias_noise_adu" in r]
    short = out.read("short")

    if bias:
        offsets = sorted({r["offset"] for r in bias})
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for o in offsets:
            rows = [r for r in bias if r["offset"] == o]
            axs[0].plot([r["gain"] for r in rows], [r["mean"] for r in rows], "o-", label=f"offset {o}")
            axs[1].plot([r["gain"] for r in rows], [r["noise_std_adu"] for r in rows], "o-", label=f"offset {o}")
            axs[2].plot([r["gain"] for r in rows], [r["clipped_frac"] for r in rows], "o-", label=f"offset {o}")
        for ax, label in zip(axs, ["bias level [ADU]", "read noise [ADU]", "clipped pixel fraction"], strict=True):
            ax.set_xlabel("gain")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[0].legend()
        fig.tight_layout()
        save(fig, "bias.png")

    if gain:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for g in sorted({r["gain"] for r in gain}):
            rows = [r for r in gain if r["gain"] == g]
            s = [r["signal_adu"] for r in rows]
            ax.plot(s, [r["variance_adu2"] for r in rows], "o", label=f"gain {g}")
            fit = fit_gain(gain, g)
            if fit:
                xs = np.linspace(0, max(s), 20)
                ax.plot(xs, fit["adu_per_e"] * xs + fit["read_noise_adu"] ** 2, "-", alpha=0.5)
        ax.set_xlabel("dark signal [ADU]")
        ax.set_ylabel("noise variance of one frame [ADU^2]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save(fig, "gain.png")

    flat = out.read("flat")
    if flat:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for g in sorted({r["gain"] for r in flat}):
            rows = [r for r in flat if r["gain"] == g]
            good = [r for r in rows if r["sat_frac"] == 0 and 0 < r["signal_adu"] <= FIT_MAX_FRACTION * FULL_SCALE]
            (line,) = axs[0].loglog(
                [max(r["signal_adu"], 0.1) for r in rows], [r["variance_adu2"] for r in rows], "o", label=f"gain {g}"
            )
            axs[1].semilogx(
                [max(r["signal_adu"], 0.1) for r in rows], [r["nonlin_pct"] for r in rows], "o-", color=line.get_color()
            )
            fit = fit_gain(good, g, weighted=True)
            if fit is None:
                continue
            xs = np.geomspace(min(r["signal_adu"] for r in good), max(r["signal_adu"] for r in good), 20)
            axs[0].loglog(
                xs, fit["adu_per_e"] * xs + fit["read_noise_adu"] ** 2, "-", color=line.get_color(), alpha=0.5
            )
        axs[0].set_xlabel("signal [ADU]")
        axs[0].set_ylabel("noise variance of one frame [ADU^2]")
        axs[1].set_xlabel("signal [ADU]")
        axs[1].set_ylabel("deviation from linear in exposure time [%]")
        for ax in axs:
            ax.grid(alpha=0.3)
        axs[0].legend()
        fig.tight_layout()
        save(fig, "flat.png")

    if temp:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        ts = sorted({r["target"] for r in temp})
        axs[0].plot(ts, [np.mean([r["bias_noise_adu"] for r in temp if r["target"] == t]) for t in ts], "o-")
        axs[0].set_ylabel("read noise [ADU]")
        for e in sorted({r["exposure"] for r in temp}):
            rows = [r for r in temp if r["exposure"] == e]
            axs[1].plot(
                [r["temp"] for r in rows],
                [max(r["dark_adu_per_s_mean"], 1e-6) for r in rows],
                "o-",
                label=f"{e:.0f} s",
            )
            axs[2].plot([r["temp"] for r in rows], [r["hot_frac"] for r in rows], "o-", label=f"{e:.0f} s")
        axs[1].set_yscale("log")
        axs[1].set_ylabel("dark signal, mean [ADU/s/pixel]")
        axs[2].set_yscale("log")
        axs[2].set_ylabel("hot pixel fraction")
        for ax in axs:
            ax.set_xlabel("sensor temperature [degC]" if ax is not axs[0] else "target [degC]")
            ax.grid(alpha=0.3)
        axs[1].legend()
        fig.tight_layout()
        save(fig, "temperature.png")

    if short:
        rows = [r for r in short if r["exposure"] >= SHORT_FIT_MIN_S]
        t = np.array([r["exposure"] for r in short])
        level = np.array([r["level"] for r in short])
        rate, level0 = np.polyfit([r["exposure"] for r in rows], [r["level"] for r in rows], 1)
        # bias of the same gain and offset from the bias step, if there is one
        match = [b["mean"] for b in bias if b["gain"] == short[0]["gain"] and b["offset"] == short[0]["offset"]]
        if match:
            print(f"Short exposures: exposure offset {(level0 - match[0]) / rate * 1e6:+.0f} us")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog(t * 1000, level - (match[0] if match else level0), "o-", label="measured")
        ax.loglog(t * 1000, rate * t + level0 - (match[0] if match else level0), "--", label="line from >= 1 ms")
        ax.set_xlabel("exposure time [ms]")
        ax.set_ylabel("level above bias [ADU]")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        save(fig, "short.png")

    written += export_csv(out.folder)
    for path in written:
        print(f"Wrote {path}")


def serial_number(camera: asi.Camera) -> str | None:
    """Serial number of the camera as ZWO's software shows it (16 hex digits), None if it cannot be read.

    zwoasi does not wrap ``ASIGetSerialNumber``, so it is called through ctypes.
    """

    class _SerialNumber(ctypes.Structure):
        _fields_ = [("id", ctypes.c_ubyte * 8)]

    func = getattr(asi.zwolib, "ASIGetSerialNumber", None)
    if func is None:
        return None
    sn = _SerialNumber()
    if func(camera.id, ctypes.byref(sn)) != 0:
        return None
    return "".join(f"{b:02x}" for b in sn.id)


def find_report_folder(root: Path) -> Path:
    """The single unit folder (``<model>/<serial number>``) below ``root`` that has a ``meta.json``."""
    folders = sorted(p.parent for p in root.glob("*/*/meta.json"))
    if len(folders) != 1:
        found = ", ".join(str(f) for f in folders) or "none"
        raise SystemExit(f"Expected exactly one camera folder in {root}, found: {found}. Use --folder.")
    return folders[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["bias", "gain", "temperature", "flat", "short", "readout", "pixels", "report"])
    parser.add_argument("--sdk", default="sdk/x64/libASICamera2.so")
    parser.add_argument("--out", default="characterization", help="root folder, results go to <out>/<model>/<serial>")
    parser.add_argument("--folder", type=Path, help="unit folder for the report step, if there is more than one")
    parser.add_argument("--gains", type=int, nargs="+", help="gains for the bias or gain step")
    parser.add_argument("--offsets", type=int, nargs="+", help="offsets for the bias step")
    parser.add_argument("--exposures", type=float, nargs="+", help="exposure times in s for the gain step")
    parser.add_argument("--temperatures", type=int, nargs="+", default=TEMPERATURES)
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN, help="gain for the temperature step")
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--target-temp", type=int, default=20, help="cooler target for the gain step in degC")
    parser.add_argument(
        "--reference", type=float, default=SHORT_REFERENCE_S, help="reference exposure in s, short step"
    )
    parser.add_argument("--no-cooling", action="store_true", help="gain step with the cooler off")
    args = parser.parse_args()

    if args.step == "report":
        # no camera needed, use the saved data
        make_report(Output(args.folder or find_report_folder(Path(args.out))))
        return

    cam = Camera(args.sdk)
    name = str(cam.info["Name"]).replace(" ", "_")
    out = Output(Path(args.out) / name / (cam.serial or "unknown_serial"))
    meta = {
        "info": {k: v for k, v in cam.info.items()},
        "serial": cam.serial,
        "sdk": cam.sdk,
        "controls": cam.controls,
    }
    (out.folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    try:
        if args.step == "bias":
            run_bias(cam, out, args.gains or GAINS, args.offsets or OFFSETS)
        elif args.step == "gain":
            target = None if args.no_cooling else args.target_temp
            run_gain(
                cam, out, args.gains or GAIN_TEST_GAINS, args.exposures or GAIN_TEST_EXPOSURES, args.offset, target
            )
        elif args.step == "flat":
            run_flat(cam, out, args.gains or FLAT_GAINS, args.offset)
        elif args.step == "short":
            run_short(cam, out, args.gain, args.offset, args.reference)
        elif args.step == "readout":
            run_readout(cam, out, args.gain, args.offset)
        elif args.step == "pixels":
            target = None if args.no_cooling else args.target_temp
            run_pixels(cam, out, args.gain, args.offset, target)
        elif args.step == "temperature":
            run_temperature(cam, out, args.temperatures, args.gain, args.offset)
    finally:
        cam.cam.close()


if __name__ == "__main__":
    sys.exit(main())
