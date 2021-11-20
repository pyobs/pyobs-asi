ASI module for *pyobs*
======================

ASI SDK
-------
Download the ASI SDK from https://astronomy-imaging-camera.com/software-drivers and install the libraries from the
appropriate folder in `lib/` to `/usr/local/lib`. Then copy `asi.rules` to `/etc/udev/rules.d/99-asi.rules`.


Install *pyobs-asi*
-------------------
Clone the repository:

    git clone https://github.com/pyobs/pyobs-asi.git


And install it:

    cd pyobs-asi
    pip3 install .


Configuration
-------------
The *AsiCamera* class is derived from *BaseCamera* (see *pyobs* documentation) and adds some new parameters:

    camera:
        Name of camera to acquire driver for.
    sdk:
        Path to .so file from ASI SDK.

Therefore, a basic module configuration would look like this:

    class: pyobs_aso.AsiCamera
    name: ASI camera
    camera: ZWO ASI071MC Pro

Dependencies
------------
* **pyobs** for the core funcionality. It is not included in the *requirements.txt*, so needs to be installed 
  separately.
* [zwoasi](https://github.com/stevemarple/python-zwoasi/) as a Python wrapper for the ASI SDK.
* [Astropy](http://www.astropy.org/) for FITS file handling.
* [NumPy](http://www.numpy.org/) for array handling.