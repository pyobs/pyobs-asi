import logging
import math
import threading
from datetime import datetime
import time
from astropy.io import fits
import numpy as np
import zwoasi as asi

from pyobs.interfaces import ICamera, ICameraWindow, ICameraBinning
from pyobs.modules.camera.basecamera import BaseCamera


log = logging.getLogger(__name__)


class AsiCamera(BaseCamera, ICamera, ICameraWindow, ICameraBinning):
    """A pyobs module for ASI cameras."""

    def __init__(self, camera: str, sdk: str = '/usr/local/lib/libASICamera2.so', *args, **kwargs):
        """Initializes a new AsiCamera.

        Args:
            camera: Name of camera to use.
            sdk: Path to .so file from ASI SDK.
        """
        BaseCamera.__init__(self, *args, **kwargs)

        # variables
        self._camera_name = camera
        self._sdk_path = sdk
        self._camera = None
        self._camera_info = None

        # window and binning
        self._window = None
        self._binning = None

    def open(self):
        """Open module."""
        BaseCamera.open(self)

        # init driver
        asi.init(self._sdk_path)

        # get number of cameras
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            raise ValueError('No cameras found')

        # get ID of camera
        # index() raises ValueError, if camera could not be found
        cameras_found = asi.list_cameras()
        camera_id = cameras_found.index(self._camera_name)

        # open driver
        self._camera = asi.Camera(camera_id)
        self._camera_info = self._camera.get_camera_property()

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        self._camera.disable_dark_subtract()
        self._camera.set_control_value(asi.ASI_GAIN, 150)
        self._camera.set_control_value(asi.ASI_EXPOSURE, 30000)
        self._camera.set_control_value(asi.ASI_WB_B, 99)
        self._camera.set_control_value(asi.ASI_WB_R, 75)
        self._camera.set_control_value(asi.ASI_GAMMA, 50)
        self._camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
        self._camera.set_control_value(asi.ASI_FLIP, 0)
        self._camera.set_image_type(asi.ASI_IMG_RAW16)

        # Enabling stills mode
        try:
            # Force any single exposure to be halted
            self._camera.stop_video_capture()
            self._camera.stop_exposure()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass

        # get initial window and binning
        self._binning = self._camera.get_bin()
        self._window = self._camera.get_roi()

    def close(self):
        """Close the module."""
        BaseCamera.close(self)

    def get_full_frame(self, *args, **kwargs) -> (int, int, int, int):
        """Returns full size of CCD.

        Returns:
            Tuple with left, top, width, and height set.
        """
        return 0, 0, self._camera_info['MaxWidth'], self._camera_info['MaxHeight']

    def get_window(self, *args, **kwargs) -> (int, int, int, int):
        """Returns the camera window.

        Returns:
            Tuple with left, top, width, and height set.
        """
        return self._window

    def get_binning(self, *args, **kwargs) -> (int, int):
        """Returns the camera binning.

        Returns:
            Tuple with x and y.
        """
        return self._binning, self._binning

    def set_window(self, left: float, top: float, width: float, height: float, *args, **kwargs):
        """Set the camera window.

        Args:
            left: X offset of window.
            top: Y offset of window.
            width: Width of window.
            height: Height of window.

        Raises:
            ValueError: If binning could not be set.
        """
        self._window = (left, top, width, height)
        log.info('Setting window to %dx%d at %d,%d...', width, height, left, top)

    def set_binning(self, x: int, y: int, *args, **kwargs):
        """Set the camera binning.

        Args:
            x: X binning.
            y: Y binning.

        Raises:
            ValueError: If binning could not be set.
        """
        self._binning = x
        log.info('Setting binning to %dx%d...', x, x)

    def _expose(self, exposure_time: int, open_shutter: bool, abort_event: threading.Event) -> fits.PrimaryHDU:
        """Actually do the exposure, should be implemented by derived classes.

        Args:
            exposure_time: The requested exposure time in ms.
            open_shutter: Whether or not to open the shutter.
            abort_event: Event that gets triggered when exposure should be aborted.

        Returns:
            The actual image.
        """

        # set window, divide width/height by binning
        width = int(math.floor(self._window[2]) / self._binning)
        height = int(math.floor(self._window[3]) / self._binning)
        log.info("Set window to %dx%d (binned %dx%d with %dx%d) at %d,%d.",
                 self._window[2], self._window[3], width, height, self._binning, self._binning,
                 self._window[0], self._window[1])
        self._camera.set_roi(int(self._window[0]), int(self._window[1]), width, height,
                             self._binning, asi.ASI_IMG_RAW16)

        # set some stuff
        self._change_exposure_status(ICamera.ExposureStatus.EXPOSING)
        self._camera.set_control_value(asi.ASI_EXPOSURE, exposure_time)

        # get date obs
        log.info('Starting exposure with %s shutter for %.2f seconds...',
                 'open' if open_shutter else 'closed', exposure_time / 1000.)
        date_obs = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")

        # do exposure
        self._camera.start_exposure()

        # wait for image
        while self._camera.get_exposure_status() == asi.ASI_EXP_WORKING:
            # aborted?
            if abort_event.is_set():
                self._change_exposure_status(ICamera.ExposureStatus.IDLE)
                raise ValueError('Aborted exposure.')

            # sleep a little
            abort_event.wait(0.01)

        # success?
        status = self._camera.get_exposure_status()
        if status != asi.ASI_EXP_SUCCESS:
            raise ValueError('Could not capture image: %s' % status)

        # get data
        log.info('Exposure finished, reading out...')
        self._change_exposure_status(ICamera.ExposureStatus.READOUT)
        buffer = self._camera.get_data_after_exposure()
        whbi = self._camera.get_roi_format()
        shape = [whbi[1], whbi[0]]
        data = np.frombuffer(buffer, dtype=np.uint16).reshape(shape)

        # create FITS image and set header
        hdu = fits.PrimaryHDU(data)
        hdu.header['DATE-OBS'] = (date_obs, 'Date and time of start of exposure')
        hdu.header['EXPTIME'] = (exposure_time / 1000., 'Exposure time [s]')

        # instrument and detector
        hdu.header['INSTRUME'] = (self._camera_name, 'Name of instrument')

        # binning
        hdu.header['XBINNING'] = hdu.header['DET-BIN1'] = (self._binning, 'Binning factor used on X axis')
        hdu.header['YBINNING'] = hdu.header['DET-BIN2'] = (self._binning, 'Binning factor used on Y axis')

        # window
        hdu.header['XORGSUBF'] = (self._window[0], 'Subframe origin on X axis')
        hdu.header['YORGSUBF'] = (self._window[1], 'Subframe origin on Y axis')

        # statistics
        hdu.header['DATAMIN'] = (float(np.min(data)), 'Minimum data value')
        hdu.header['DATAMAX'] = (float(np.max(data)), 'Maximum data value')
        hdu.header['DATAMEAN'] = (float(np.mean(data)), 'Mean data value')

        # pixels
        hdu.header['DET-PIXL'] = (self._camera_info['PixelSize'] / 1000., 'Size of detector pixels (square) [mm]')
        hdu.header['DET-GAIN'] = (self._camera_info['ElecPerADU'], 'Detector gain [e-/ADU]')

        # biassec/trimsec
        self.set_biassec_trimsec(hdu.header, *self._window)

        # return FITS image
        log.info('Readout finished.')
        self._change_exposure_status(ICamera.ExposureStatus.IDLE)
        return hdu

    def _abort_exposure(self):
        """Abort the running exposure. Should be implemented by derived class.

        Raises:
            ValueError: If an error occured.
        """
        pass


__all__ = ['AsiCamera']
