*pyobs* for ZWO ASI cameras
===========================

This is a [pyobs](https://www.pyobs.org) module for [ZWO ASI](https://astronomy-imaging-camera.com/) cameras.


ASI SDK
-------
Download the ASI SDK from https://astronomy-imaging-camera.com/software-drivers and install the libraries from the
appropriate folder in `lib/` to `/usr/local/lib`. Then copy `asi.rules` to `/etc/udev/rules.d/99-asi.rules`.


Install *pyobs-asi*
--------------------
Clone the repository:

    git clone https://github.com/pyobs/pyobs-asi.git
    cd pyobs-asi

Install it with [uv](https://docs.astral.sh/uv/):

    uv sync

Alternatively, with plain `venv`/`pip`:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install .


Configuration
-------------
The *AsiCamera* class is derived from *BaseCamera* (see *pyobs* documentation) and adds some new parameters:

    camera:
        Name of camera to acquire driver for.
    sdk:
        Path to .so file from ASI SDK.

Therefore, a basic module configuration would look like this:

    class: pyobs_asi.AsiCamera
    name: ASI camera
    camera: ZWO ASI071MC Pro

For cameras with cooling, use *AsiCoolCamera* instead, which adds a `setpoint` parameter for the initial cooling
setpoint in degrees Celsius:

    class: pyobs_asi.AsiCoolCamera
    name: ASI camera
    camera: ZWO ASI071MC Pro
    setpoint: -20


GUI
---
For testing a camera without a full *pyobs* setup, install the optional `gui` extra:

    uv sync --extra gui

and run:

    uv run asi-gui [path/to/libASICamera2.so]


Dependencies
------------
* [pyobs-core](https://github.com/pyobs/pyobs-core) for the core functionality.
* [zwoasi](https://github.com/stevemarple/python-zwoasi/) as a Python wrapper for the ASI SDK.
* [Astropy](http://www.astropy.org/) for FITS file handling.
* [NumPy](http://www.numpy.org/) for array handling.