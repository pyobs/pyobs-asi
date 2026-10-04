# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries for releases before this file existed were generated from commit subjects.

## [2.1.1] - 2026-10-04

- Use __future__ annotations instead of Optional (ruff UP045)
- Fix import on Python 3.11/3.12 (Optional[threading.Lock]), test 3.11-3.13 in CI

## [2.1.0] - 2026-09-29

- Add AsiVideo module implementing IVideo via BaseVideo.frames()

## [2.0.4] - 2026-09-28

- Maintenance release (dependency and metadata updates only).

## [2.0.3] - 2026-09-28

- Add IResettable.reset()/full_reset() overrides

## [2.0.2] - 2026-09-08

- fix: publish AsiCamera capabilities after connecting, not before

## [2.0.1] - 2026-09-03

- Add OFFSET/DET-COOL/DET-TSET FITS headers (#872)

## [2.0.0] - 2026-08-26

- Require stable pyobs-core>=2.0.0
- Gate auto-merge on the PR author, not the event actor
- Enable Dependabot auto-merge for patch/minor updates
- Camera driver/GUI split: abort stop_exposure, SDK locking, gui offload (#32)
- tests: assert comm.set_state call shape in window/binning/gain/offset tests
- pyrefly: exclude gui.py from type checking
- Add baseline test suite and CI (pytest, pyrefly), grouped Dependabot
- Upgrade uv.lock to clear open Dependabot alerts
- Require pyobs-core>=2.0.0.dev48
- Add dependabot.yml, targeting develop for PRs
- Run ZWO ASI SDK calls through a background thread instead of the event loop
- Use pyobs-core[gui] extra instead of separate Qt packages
- Fix Sphinx docs to reflect both camera classes
- Rewrite README for pyobs 2.0 / uv workflow
- Add GUI script for ASI cameras
- migrated to Ruff and updated modules to pyobs 2.0 API
- removed 2nd numpy dep
- new lock file

## [1.1.0] - 2025-07-07

- migrated to uv
- datetime.utcnow() to datetime.utc(timezone.utc)

## [1.0.3] - 2024-03-21

- fixed docs

## [1.0.2] - 2024-02-20

- Changed highest compatible python version to 3.11
- Added and implemented ITemperatures AsiCamera
- Added IGain

## [1.0.1] - 2023-03-24

- added AsiCoolCamera to docs

## [1.0.0] - 2022-09-13

- added license

## [0.20.0] - 2022-07-20

- upgrade to pyobs-core >0.20
- added IAbortable
- renamed get_cooling_status to get_cooling
- example config
- added file

## [0.16.0] - 2022-01-18

- fixed rtd
- basic docs
- set InterruptedError instead of AbortedError
- replaced AbortedError with builtin InterruptedError
- new exceptions for cameras
- added black and pre-commit to dev dependencies
- added .pre-commit-config.yaml
- running black
- added black config

## [0.15.0] - 2021-12-29

- changed used Python version to 3.9
- Pushed requirements to Python>=3.9 and astropy>=5.0, closes #55
- moved image format methods from AsiCoolCamera to AsiCamera
- cleaning up
- asyncio
- added github action
- switched to poetry
- v0.14
- temp
- temperature
- disable cooling
- Added type hints
- renamed ICameraWindow to IWindow
- renamed ICameraBinning to IBinning
- v0.13
- documentation
- added __module__
- updated docstrings
- moved ExposureStatus to utils.enums
- fixed type hints
- fixed type hint and made mypy happy
- Moved "images" module to top-level
- writing BAYERPAT and COLORTYP to fits header
- changed treatment of RGB images
- changed signature of ICameraBinning's list_binnings() to return list of X,Y tuples
- implemented list_binnings()
- fixed list of formats
- changed dimension
- _expose() should return an Image
- swapped bit lengths
- fixed bug
- color mode
- new enum for image formats
- fixed exptime in fits headers
- asi expects exposure time in micro seconds
- initial sleep after start_exposure
- unit of exptime
- added support for cooled cameras
- fixed typo
- added DET-PIXL and DET-GAIN to FITS header
- initial commit
