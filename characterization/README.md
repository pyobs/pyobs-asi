# Camera characterization

Measurements of ASI cameras run with pyobs-asi. Each camera model has its own folder with notes that apply to every
unit of that model, and one subfolder per unit, named after its serial number, with the measurements.

| Model | Units |
|---|---|
| [ZWO ASI2600MM Pro](ZWO_ASI2600MM_Pro/README.md) | [1d22d2002b020900](ZWO_ASI2600MM_Pro/1d22d2002b020900/README.md) |

## Layout

```
characterization/
  <model>/
    README.md          notes for the model: SDK, driver, recommended settings
    <serial number>/
      README.md        report of the measurements on this unit
      *.jsonl          raw records written by tools/characterize.py
      *.csv            the same data as tables, plus results of separate analyses
      meta.json        camera properties and controls as reported by the SDK, and the serial number
      images/          plots
      frames/          512x512 FITS cutouts (not kept in git)
```

The serial number is the one ZWO's software shows (`ASIGetSerialNumber` in the SDK, 16 hexadecimal digits).

## Reproduce

`tools/characterize.py` runs the measurements (no other program may use the camera). It writes into
`characterization/<model>/<serial number>/` of the connected camera. With the lens capped:

```bash
uv run python tools/characterize.py bias
uv run python tools/characterize.py gain --target-temp 10
uv run python tools/characterize.py temperature
uv run python tools/characterize.py pixels --no-cooling
uv run python tools/characterize.py pixels --target-temp -5
uv run python tools/characterize.py readout
```

With the lens open, pointed at a steady, defocused light source:

```bash
uv run python tools/characterize.py flat
uv run python tools/characterize.py flat --gains 25 75 150 250
uv run python tools/characterize.py short --reference 0.025
```

The `report` step needs no camera. It redraws the plots in `images/` and writes a CSV file next to every `.jsonl`
file. With more than one unit measured, give the folder:

```bash
uv run python tools/characterize.py report --folder characterization/ZWO_ASI2600MM_Pro/1d22d2002b020900
```

Not produced by the script: the 17 h dark of the ASI2600MM Pro and its results (`dark_long_*.csv`,
`bias_long_frames.csv`, `hotpixels_-10C.csv`, `images/dark_long.png`). The frames were taken with `asi-gui` and
evaluated separately.
