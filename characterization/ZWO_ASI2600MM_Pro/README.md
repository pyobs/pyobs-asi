# ZWO ASI2600MM Pro

Notes on the ZWO ASI2600MM Pro (monochrome, cooled CMOS camera) that apply to every unit of this model when driven by
pyobs-asi: SDK and driver behaviour and recommended settings. The recommendations are based on the measurements of one
unit so far. The measurements are per unit:

| Serial number | Measured | Report |
|---|---|---|
| 1d22d2002b020900 | November 2026 | [1d22d2002b020900/README.md](1d22d2002b020900/README.md) |

The serial number is the one ZWO's software shows (`ASIGetSerialNumber` in the SDK).

## Camera properties

Camera properties as reported by the SDK (from the SDK, full list in
[meta.json](1d22d2002b020900/meta.json)):

| Property | Value |
|---|---|
| Sensor | 6248 x 4176 pixels of 3.76 µm, mono, 16 bit |
| Cooler, shutter | cooled, no mechanical shutter |
| Binning | 1, 2, 3, 4 |
| Exposure | 32 µs to 2000 s |
| Gain | -25 to 700 (default 200) |
| Offset | 0 to 240 (default 1) |
| `ElecPerADU` | 0.243 (close to the 0.255 e-/ADU measured at gain 100) |

## Recommended use

- Use gain 100 and offset 50 (the pyobs-asi defaults), unless you need the full well of gain 0 (51 ke- instead of
  17 ke-). Gain 100 is the lowest tested gain with the low read noise of the high conversion gain mode.
- Never use offset 0: about half of the pixels are clipped at zero at every gain. Offset 50 is clean up to gain 300.
- Keep high speed mode off.
- Take bias and dark frames only once the sensor temperature has settled (within about 0.1 °C), and drop the first
  frames of a long dark series (the first four were warm in the
  [17 h dark](1d22d2002b020900/README.md#long-dark-at--10-c-measured)).
- Subtract a bias or dark. That removes the fixed row pattern and the hot pixels. Dither and use sigma-clipped or
  median stacking against the salt-and-pepper noise (inferred, not tested).
- For exposures below a few ms (flat fields, bright stars) add about 92 µs to the set exposure time (measured at gain
  100 and 28 °C only).
- Choose a cooling target at least 5 °C below ambient, and not more than about 30 °C below it.

## Software notes

**SDK (measured).** A community copy of SDK 1.37 returned corrupt frames (flat ~58000 ADU, all zeros, or 14% wrong rows
with high speed mode). The official SDK 1.41 gives clean frames. With high speed mode on, 1.41 still gives about 4 odd
pixels per row (presumably edge columns, not checked); `asi-gui` therefore switches it off on startup. The SDK resets the
USB device when the camera is opened and occasionally during use (`reset SuperSpeed USB device` in the kernel log);
there are no transfer errors, so these do not seem to be faults (inferred).

**SDK constraints (from the SDK).** The window width must be a multiple of 8 and the height a multiple of 2, in binned
pixels, and the window start is given in binned pixels too. The shortest exposure is 32 µs; 0 s gives garbage frames.

**Driver (`AsiCamera`, `AsiCoolCamera`), measured on the camera:**

- Gain modes: "Low noise" (gain 100) and "High capacity" (gain 0), written to the FITS header with `DET-GAIN`, `GAIN`
  and `OFFSET`.
- Fixed in pyobs-asi 2.3.0 ([pyobs/pyobs-asi#81](https://github.com/pyobs/pyobs-asi/pull/81)): a cooling ramp
  started from 0 °C because of the 0.0 °C reading after opening
  ([#79](https://github.com/pyobs/pyobs-asi/issues/79)); the warm-up on close stopped after a fixed 300 s timeout
  ([#80](https://github.com/pyobs/pyobs-asi/issues/80)); the `GAIN` keyword was missing.
- Fixed in pyobs-core 2.27.1 ([pyobs/pyobs-core#1077](https://github.com/pyobs/pyobs-core/pull/1077)): the warm-up
  switched the cooler off after the first step, because the sensor lags behind the setpoint. It now also waits for zero
  cooler power.
- Full cycle with these versions: cool from 28 °C to -5 °C in about 17 min, hold at 54% to 60% power, warm up and close
  in 600 s, with the sensor within about 1 °C of the setpoint during the warm-up.
- Fixed for the next release: with binning above 1 and a window not starting at 0, the window was placed at the
  wrong position (the start was given in unbinned instead of binned pixels).

**GUI (`asi-gui`):** gain and offset, cooling with the same ramps as the driver, sensor temperature and cooler power
every 2 s, exposure time and window rounded to what the SDK accepts. From the next release the saved frames carry a
FITS header (time, exposure time, gain, offset, sensor temperature, setpoint, cooler power, binning, window, pixel
size), with the same keywords as the pyobs driver. Earlier versions saved frames without a header.

