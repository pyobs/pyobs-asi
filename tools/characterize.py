"""Characterize a cooled ASI camera in the dark.

Run with the lens capped and without any other program using the camera, e.g.::

    uv run python tools/characterize.py bias --sdk sdk/x64/libASICamera2.so
    uv run python tools/characterize.py gain
    uv run python tools/characterize.py temperature
    uv run python tools/characterize.py flat   # lens uncapped, pointed at a steady, defocused light source
    uv run python tools/characterize.py report

Steps (all with high speed mode off, on a central 2048x2048 window):

* ``bias``: bias level, read noise and clipping vs. gain and offset.
* ``gain``: gain in ADU/e- from the shot noise of the dark current (photon transfer without light). Needs a warm
  sensor, so it turns the cooler off and waits until the temperature is stable.
* ``temperature``: read noise, dark current and hot pixels vs. sensor temperature. The cooler target is moved in steps
  of at most 2 degC/min and the camera is warmed up again afterwards.
* ``flat``: photon transfer curve from pairs of frames of a steady, roughly uniform light source (e.g. a wall): gain in
  e-/ADU, read noise in e-, saturation and linearity. The bias level comes from the fit of level vs. exposure time.
* ``report``: plots and a Markdown report from the measured data.

Results are appended to JSON lines files in the output folder, so a run can be repeated or resumed.
"""

import argparse
import json
import math
import sys
import time
from collections import deque
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


def make_report(out: Output, info: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out.folder
    lines = [f"# Characterization of {info.get('Name', 'ASI camera')}", ""]
    bias, gain, temp = out.read("bias"), out.read("gain"), out.read("temperature")
    temp = [r for r in temp if "bias_noise_adu" in r]

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
        fig.savefig(folder / "bias.png", dpi=110)
        lines += ["## Bias vs gain and offset", "", "![bias](bias.png)", ""]
        lines += ["| gain | offset | bias [ADU] | read noise [ADU] | clipped |", "|---|---|---|---|---|"]
        for r in bias:
            lines.append(
                f"| {r['gain']} | {r['offset']} | {r['mean']:.1f} | {r['noise_std_adu']:.2f} | "
                f"{r['clipped_frac']:.4f} |"
            )
        lines.append("")

    if gain:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        lines += [
            "## Gain from dark-current shot noise",
            "",
            "| gain | e-/ADU | ADU/e- | slope error | read noise [ADU] |",
        ]
        lines += ["|---|---|---|---|---|"]
        for g in sorted({r["gain"] for r in gain}):
            rows = [r for r in gain if r["gain"] == g]
            s = [r["signal_adu"] for r in rows]
            ax.plot(s, [r["variance_adu2"] for r in rows], "o", label=f"gain {g}")
            fit = fit_gain(gain, g)
            if fit:
                xs = np.linspace(0, max(s), 20)
                ax.plot(xs, fit["adu_per_e"] * xs + fit["read_noise_adu"] ** 2, "-", alpha=0.5)
                lines.append(
                    f"| {g} | {fit['e_per_adu']:.3f} | {fit['adu_per_e']:.3f} | {fit['slope_err']:.3f} | "
                    f"{fit['read_noise_adu']:.2f} |"
                )
        ax.set_xlabel("dark signal [ADU]")
        ax.set_ylabel("noise variance of one frame [ADU^2]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / "gain.png", dpi=110)
        lines += ["", "![gain](gain.png)", ""]
        lines.append(f"SDK value at open: ElecPerADU = {info.get('ElecPerADU')}")
        lines.append("")

    flat = out.read("flat")
    if flat:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        lines += [
            "## Photon transfer from a light source",
            "",
            f"Fit below {FIT_MAX_FRACTION:.0%} of full scale, central {CUTOUT}x{CUTOUT} pixels, pairs of frames.",
            "",
            "| gain | e-/ADU | ADU/e- | slope error | read noise [ADU] | read noise [e-] | full scale [e-] | "
            "max nonlinearity [%] |",
            "|---|---|---|---|---|---|---|---|",
        ]
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
            bias = rows[0]["bias_fit"]
            nonlin = max(abs(r["nonlin_pct"]) for r in good)
            err = fit["slope_err"] / fit["adu_per_e"]
            lines.append(
                f"| {g} | {fit['e_per_adu']:.3f} | {fit['adu_per_e']:.3f} | {err:.1%} | "
                f"{fit['read_noise_adu']:.2f} | {fit['read_noise_adu'] * fit['e_per_adu']:.2f} | "
                f"{(FULL_SCALE - bias) * fit['e_per_adu']:.0f} | {nonlin:.2f} |"
            )
        axs[0].set_xlabel("signal [ADU]")
        axs[0].set_ylabel("noise variance of one frame [ADU^2]")
        axs[1].set_xlabel("signal [ADU]")
        axs[1].set_ylabel("deviation from linear in exposure time [%]")
        for ax in axs:
            ax.grid(alpha=0.3)
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(folder / "flat.png", dpi=110)
        lines += ["", "![flat](flat.png)", ""]

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
        fig.savefig(folder / "temperature.png", dpi=110)
        lines += ["## Noise and dark current vs temperature", "", "![temperature](temperature.png)", ""]
        lines += ["| T [degC] | exposure [s] | read noise [ADU] | dark [ADU/s] mean | hot fraction |"]
        lines += ["|---|---|---|---|---|"]
        for r in temp:
            lines.append(
                f"| {r['temp']:.1f} | {r['exposure']:.0f} | {r['bias_noise_adu']:.2f} | "
                f"{r['dark_adu_per_s_mean']:.5f} | {r['hot_frac']:.5f} |"
            )
        lines.append("")

    (folder / "report.md").write_text("\n".join(lines))
    print(f"Wrote {folder / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["bias", "gain", "temperature", "flat", "report"])
    parser.add_argument("--sdk", default="sdk/x64/libASICamera2.so")
    parser.add_argument("--out", default="characterization")
    parser.add_argument("--gains", type=int, nargs="+", help="gains for the bias or gain step")
    parser.add_argument("--offsets", type=int, nargs="+", help="offsets for the bias step")
    parser.add_argument("--exposures", type=float, nargs="+", help="exposure times in s for the gain step")
    parser.add_argument("--temperatures", type=int, nargs="+", default=TEMPERATURES)
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN, help="gain for the temperature step")
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--target-temp", type=int, default=20, help="cooler target for the gain step in degC")
    parser.add_argument("--no-cooling", action="store_true", help="gain step with the cooler off")
    args = parser.parse_args()

    if args.step == "report":
        # no camera needed, use the saved data
        folders = [f for f in Path(args.out).iterdir() if (f / "meta.json").exists()]
        if len(folders) != 1:
            raise SystemExit(f"Expected exactly one camera folder in {args.out}, found {len(folders)}.")
        info = json.loads((folders[0] / "meta.json").read_text())["info"]
        make_report(Output(folders[0]), info)
        return

    cam = Camera(args.sdk)
    name = str(cam.info["Name"]).replace(" ", "_")
    out = Output(Path(args.out) / name)
    meta = {"info": {k: v for k, v in cam.info.items()}, "sdk": cam.sdk, "controls": cam.controls}
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
        elif args.step == "temperature":
            run_temperature(cam, out, args.temperatures, args.gain, args.offset)
    finally:
        cam.cam.close()


if __name__ == "__main__":
    sys.exit(main())
