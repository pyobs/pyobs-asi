pyobs-asi
#########

This is a `pyobs <https://www.pyobs.org>`_ (`documentation <https://docs.pyobs.org>`_) module for ASI ZWO cameras.


ASI SDK
*******

The ASI SDK 1.41 libraries for Linux (``armv6``, ``armv7``, ``armv8``, ``x64``, ``x86``) and the udev rules
(``asi.rules``) are included in the ``sdk/`` folder of the repository, as published by ZWO under the license in
``sdk/license.txt``. See the README for installation, or point the ``sdk`` parameter to the library for your platform,
e.g. ``sdk/x64/libASICamera2.so``. Newer versions are available at https://www.zwoastro.com/software/product-sdk/.


Example configuration
*********************

This is an example configuration, tested on a ASI 071 MC::

    class: pyobs_asi.AsiCoolCamera
    camera: ZWO ASI071MC Pro

    # optional: path to the ASI SDK library, if it is not installed system-wide
    sdk: /path/to/pyobs-asi/sdk/x64/libASICamera2.so

    # file naming
    filenames: /cache/pyobs-{DAY-OBS|date:}-{FRAMENUM|string:04d}-{IMAGETYP|type}00.fits

    # additional fits headers
    fits_headers:
      'DET-PIXL': [0.00478, 'Size of detector pixels (square) [mm]']
      'DET-NAME': ['SONY IMX071', 'Name of detector']
      'DET-RON': [2.3, 'Detector readout noise [e-]']
      'DET-SATU': [46000, 'Detector saturation limit [e-]']

    # opto-mechanical centre
    centre: [2472.0, 1642.0]

    # rotation (east of north)
    rotation: 3.06
    flip: True

    # location
    timezone: utc
    location:
      longitude: 9.944333
      latitude: 51.560583
      elevation: 201.

    # communication
    comm:
      jid: test@example.com
      password: ***

    # virtual file system
    vfs:
      class: pyobs.vfs.VirtualFileSystem
      roots:
        cache:
          class: pyobs.vfs.HttpFile
          upload: http://localhost:37075/


Available classes
*****************

There are two classes for ASI ZWO cameras: :class:`~pyobs_asi.AsiCamera` for cameras without cooling, and
:class:`~pyobs_asi.AsiCoolCamera` for cameras with cooling support. :class:`~pyobs_asi.AsiVideo` streams
video from an ASI camera instead, e.g. for acquisition, guiding or focusing. Only one of them can use a camera at a
time.

AsiCamera
=========
.. autoclass:: pyobs_asi.AsiCamera
   :members:
   :show-inheritance:

AsiCoolCamera
=============
.. autoclass:: pyobs_asi.AsiCoolCamera
   :members:
   :show-inheritance:

AsiVideo
========
.. autoclass:: pyobs_asi.AsiVideo
   :members:
   :show-inheritance:
