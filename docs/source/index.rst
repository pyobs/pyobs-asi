pyobs-asi
#########

This is a `pyobs <https://www.pyobs.org>`_ (`documentation <https://docs.pyobs.org>`_) module for ASI ZWO cameras.


Example configuration
*********************

This is an example configuration, tested on a ASI 071 MC::

    class: pyobs_asi.AsiCoolCamera
    camera: ZWO ASI071MC Pro

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

There is one single class for ASI ZWO cameras.

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