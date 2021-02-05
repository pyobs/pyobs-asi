import logging
import math
import threading
from datetime import datetime
import time
from astropy.io import fits
import numpy as np
import zwoasi as asi

from pyobs.interfaces import ICamera, ICameraWindow, ICameraBinning, ICooling, IImageFormat
from pyobs.modules.camera.basecamera import BaseCamera
from pyobs.utils.enums import ImageFormat

log = logging.getLogger(__name__)


# map of image formats
FORMATS = {
    ICameraMode.Mode.INT8: asi.ASI_IMG_RAW8,
    ICameraMode.Mode.INT16: asi.ASI_IMG_RAW16,
    ICameraMode.Mode.RGB24: asi.ASI_IMG_RGB24
}


class AsiCamera(BaseCamera, ICamera, ICameraWindow, ICameraBinning, IImageFormat):
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

        # window and binning and mode
        self._window = None
        self._binning = None
        self._image_format = ImageFormat.INT16

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
        log.info('Camera info:')
        for key, val in self._camera_info.items():
            log.info('  - %s: %s', key, val)

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        self._camera.disable_dark_subtract()
        self._camera.set_control_value(asi.ASI_GAIN, 150)
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

    def _expose(self, exposure_time: float, open_shutter: bool, abort_event: threading.Event) -> fits.PrimaryHDU:
        """Actually do the exposure, should be implemented by derived classes.

        Args:
            exposure_time: The requested exposure time in s.
            open_shutter: Whether or not to open the shutter.
            abort_event: Event that gets triggered when exposure should be aborted.

        Returns:
            The actual image.
        """

        # get image format
        image_format = FORMATS[self._image_format]

        # set window, divide width/height by binning
        width = int(math.floor(self._window[2]) / self._binning)
        height = int(math.floor(self._window[3]) / self._binning)
        log.info("Set window to %dx%d (binned %dx%d with %dx%d) at %d,%d.",
                 self._window[2], self._window[3], width, height, self._binning, self._binning,
                 self._window[0], self._window[1])
        self._camera.set_roi(int(self._window[0]), int(self._window[1]), width, height,
                             self._binning, image_format)

        # set status and exposure time in ms
        self._change_exposure_status(ICamera.ExposureStatus.EXPOSING)
        self._camera.set_control_value(asi.ASI_EXPOSURE, int(exposure_time * 1e6))

        # get date obs
        log.info('Starting exposure with %s shutter for %.2f seconds...',
                 'open' if open_shutter else 'closed', exposure_time)
        date_obs = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")

        # do exposure
        self._camera.start_exposure()
        self.closing.wait(0.01)

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

        # decide on image format
        if image_format == asi.ASI_IMG_RAW8:
            shape = [whbi[1], whbi[0]]
            data = np.frombuffer(buffer, dtype=np.uint16).reshape(shape)
        elif image_format == asi.ASI_IMG_RAW8:
            shape = [whbi[1], whbi[0]]
            data = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
        elif image_format == asi.ASI_IMG_RGB24:
            shape = [whbi[1], whbi[0], 3]
            data = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
        else:
            raise ValueError('Unknown image format.')

        # create FITS image and set header
        hdu = fits.PrimaryHDU(data)
        hdu.header['DATE-OBS'] = (date_obs, 'Date and time of start of exposure')
        hdu.header['EXPTIME'] = (exposure_time, 'Exposure time [s]')

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


class AsiCoolCamera(AsiCamera, ICooling):
    """A pyobs module for ASI cameras with cooling."""

    def __init__(self, setpoint: int = -20, *args, **kwargs):
        """Initializes a new AsiCoolCamera.

        Args:
            setpoint: Cooling temperature setpoint.
        """
        AsiCamera.__init__(self, *args, **kwargs)

        # variables
        self._temp_setpoint = setpoint

    def open(self):
        """Open module."""
        AsiCamera.open(self)

        # no cooling support?
        if not self._camera_info['IsCoolerCam']:
            raise ValueError('Camera has no support for cooling.')

        # activate cooling
        self.set_cooling(True, self._temp_setpoint)

    def get_cooling_status(self, *args, **kwargs) -> (bool,  float, float):
        """Returns the current status for the cooling.

        Returns:
            Tuple containing:
                Enabled (bool):         Whether the cooling is enabled
                SetPoint (float):       Setpoint for the cooling in celsius.
                Power (float):          Current cooling power in percent or None.
        """
        enabled = self._camera.get_control_value(asi.ASI_COOLER_ON)[0]
        temp = self._camera.get_control_value(asi.ASI_TARGET_TEMP)[0]
        power = self._camera.get_control_value(asi.ASI_COOLER_POWER_PERC)[0]
        return enabled, temp, power

    def get_temperatures(self, *args, **kwargs) -> dict:
        """Returns all temperatures measured by this module.

        Returns:
            Dict containing temperatures.
        """
        return {
            'CCD': self._camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.
        }

    def set_cooling(self, enabled: bool, setpoint: float, *args, **kwargs):
        """Enables/disables cooling and sets setpoint.

        Args:
            enabled: Enable or disable cooling.
            setpoint: Setpoint in celsius for the cooling.

        Raises:
            ValueError: If cooling could not be set.
        """

        # log
        if enabled:
            log.info('Enabling cooling with a setpoint of %.2f°C...', setpoint)
            self._camera.set_control_value(asi.ASI_TARGET_TEMP, int(setpoint))
            self._camera.set_control_value(asi.ASI_COOLER_ON, 1)
        else:
            log.info('Disabling cooling...')
            self._camera.set_control_value(asi.ASI_COOLER_ON, 1)

    def set_image_format(self, format: ImageFormat, *args, **kwargs):
        """Set the camera image format.

        Args:
            format: New image format.

        Raises:
            ValueError: If format could not be set.
        """
        if format not in FORMATS:
            raise ValueError('Unsupported image format.')
        self._image_format = format

    def get_image_format(self, *args, **kwargs) -> ImageFormat:
        """Returns the camera image format.

        Returns:
            Current image format.
        """
        return self._image_format

    def list_image_formats(self, *args, **kwargs) -> list:
        """List available image formats.

        Returns:
            List of available image formats.
        """
        return list(FORMATS.keys())


__all__ = ['AsiCamera', 'AsiCoolCamera']
