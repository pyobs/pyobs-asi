import asyncio
import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import numpy as np
import zwoasi as asi  # type: ignore
from numpy.typing import NDArray
from pyobs.images import Image
from pyobs.interfaces import (
    Binning,
    BinningCapabilities,
    BinningState,
    CoolingState,
    GainState,
    IBinning,
    ICooling,
    IGain,
    IImageFormat,
    ImageFormatCapabilities,
    ImageFormatState,
    ITemperatures,
    IWindow,
    SensorReading,
    TemperaturesState,
    WindowCapabilities,
    WindowState,
)
from pyobs.modules.camera.basecamera import BaseCamera
from pyobs.utils import exceptions as exc
from pyobs.utils.enums import ExposureStatus, ImageFormat

log = logging.getLogger(__name__)

_T = TypeVar("_T")

FORMATS = {
    ImageFormat.INT8: asi.ASI_IMG_RAW8,
    ImageFormat.INT16: asi.ASI_IMG_RAW16,
    ImageFormat.RGB24: asi.ASI_IMG_RGB24,
}

# ZWO ASI SDK calls are blocking and are made directly on the event loop thread (see
# _run_blocking). If the camera has gone unresponsive, they can hang indefinitely, so they're
# bounded with a timeout rather than let a single dead camera freeze the whole module.
_SDK_CALL_TIMEOUT = 5.0

# reading out a frame after exposure can take longer than the other SDK calls above, depending on
# sensor resolution/image format.
_READOUT_TIMEOUT = 30.0

# the exposure-wait loop's own timeout needs to cover the actual exposure time plus some margin
# for overhead -- exposure_time alone would be too tight.
_EXPOSURE_WAIT_MARGIN = 30.0


class AsiCamera(BaseCamera, IWindow, IBinning, IImageFormat, IGain, ITemperatures):
    """A pyobs module for ASI cameras."""

    __module__ = "pyobs_asi"

    def __init__(self, camera: str, sdk: str = "/usr/local/lib/libASICamera2.so", **kwargs: Any):
        """Initializes a new AsiCamera.

        Args:
            camera: Name of camera to use.
            sdk: Path to .so file from ASI SDK.
        """
        BaseCamera.__init__(self, **kwargs)

        self._camera_name = camera
        self._sdk_path = sdk
        self._camera: asi.Camera | None = None
        self._camera_info: dict[str, Any] = {}

        self._window = (0, 0, 0, 0)
        self._binning = 1
        self._image_format = ImageFormat.INT16

        self._gain: float = 1.0
        self._gain_offset: float = 50.0

        # The ASI SDK is not thread-safe: the 5s temperature poll and in-progress exposure
        # threads would otherwise hit the same camera handle concurrently.
        self._sdk_lock = threading.Lock()

        self.add_background_task(self._temperature_thread, True)

    @staticmethod
    async def _run_blocking(
        func: Callable[[], None], timeout: float = _SDK_CALL_TIMEOUT, lock: threading.Lock | None = None
    ) -> bool:
        """Run a blocking ASI SDK call in a daemon thread, so a hung call can't freeze the module.

        A plain executor isn't used here, since its worker threads are non-daemon and Python joins
        them on interpreter shutdown -- a hung call would then just move the freeze to process exit.

        Args:
            func: The blocking call to run.
            timeout: How long to wait for func to complete.
            lock: Optional lock to hold while func runs, guarding SDK access against other threads.

        Returns:
            True if func completed within timeout, False if it's still running in the background.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _wrapper() -> None:
            try:
                if lock is not None:
                    with lock:
                        func()
                else:
                    func()
            finally:
                loop.call_soon_threadsafe(future.set_result, None)

        threading.Thread(target=_wrapper, daemon=True).start()
        try:
            await asyncio.wait_for(future, timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def _run_blocking_or_raise(
        self, func: Callable[[], _T], timeout: float = _SDK_CALL_TIMEOUT, lock: threading.Lock | None = None
    ) -> _T:
        """Run a blocking ASI SDK call in a thread, returning its result or re-raising what it raised.

        Unlike _run_blocking(), this also carries the callable's return value/exception back to the
        caller -- several ASI calls here drive control flow via their return value or a raised
        ValueError (e.g. camera lookup by name, exposure status), which a bare fire-and-forget
        thread call would otherwise silently lose.

        Args:
            func: The blocking call to run.
            timeout: How long to wait for func to complete.
            lock: Optional lock to hold while func runs, guarding SDK access against other threads.
        """
        outcome: list[Any] = []

        def _wrapper() -> None:
            try:
                outcome.append(func())
            except BaseException as e:
                outcome.append(e)

        if not await self._run_blocking(_wrapper, timeout=timeout, lock=lock):
            raise TimeoutError(f"Timed out waiting for ASI SDK call after {timeout}s.")
        value = outcome[0]
        if isinstance(value, BaseException):
            raise value
        return cast(_T, value)

    async def open(self) -> None:
        """Open module."""

        def _open() -> tuple[dict[str, Any], int, tuple[int, int, int, int]]:
            asi.init(self._sdk_path)

            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                raise ValueError("No cameras found")

            # index() raises ValueError, if camera could not be found
            cameras_found = asi.list_cameras()
            camera_id = cameras_found.index(self._camera_name)

            self._camera = asi.Camera(camera_id)
            camera_info = self._camera.get_camera_property()

            # Set some sensible defaults. They will need adjusting depending upon
            # the sensitivity, lens and lighting conditions used.
            self._camera.disable_dark_subtract()
            self._camera.set_control_value(asi.ASI_WB_B, 99)
            self._camera.set_control_value(asi.ASI_WB_R, 75)
            self._camera.set_control_value(asi.ASI_GAMMA, 50)
            self._camera.set_control_value(asi.ASI_FLIP, 0)
            self._camera.set_image_type(asi.ASI_IMG_RAW16)

            # enabling image mode
            self._camera.stop_video_capture()
            self._camera.stop_exposure()

            # get initial window and binning
            binning = self._camera.get_bin()
            left, top, width, height = self._camera.get_roi()
            return camera_info, binning, (left, top, width, height)

        self._camera_info, self._binning, self._window = await self._run_blocking_or_raise(_open, lock=self._sdk_lock)

        log.info("Camera info:")
        for key, val in self._camera_info.items():
            log.info("  - %s: %s", key, val)

        # publish capabilities before super().open()
        await self.comm.set_capabilities(
            IWindow,
            WindowCapabilities(
                full_frame_x=0,
                full_frame_y=0,
                full_frame_width=self._camera_info["MaxWidth"],
                full_frame_height=self._camera_info["MaxHeight"],
            ),
        )
        if "SupportedBins" in self._camera_info:
            await self.comm.set_capabilities(
                IBinning,
                BinningCapabilities(binnings=[Binning(x=b, y=b) for b in self._camera_info["SupportedBins"]]),
            )
        await self.comm.set_capabilities(IImageFormat, ImageFormatCapabilities(image_formats=list(FORMATS.keys())))

        await BaseCamera.open(self)

        # publish initial states
        await self.comm.set_state(IWindow, WindowState(*self._window))
        await self.comm.set_state(IBinning, BinningState(x=self._binning, y=self._binning))
        await self.comm.set_state(IGain, GainState(gain=self._gain, offset=self._gain_offset))
        await self.comm.set_state(IImageFormat, ImageFormatState(image_format=self._image_format))

    async def _temperature_thread(self) -> None:
        """Periodically publishes temperature (and cooling) readings."""
        while True:
            if self._camera is not None:
                await self._publish_temperatures()
            await asyncio.sleep(5)

    async def _publish_temperatures(self) -> None:
        await self.comm.set_state(
            ITemperatures,
            TemperaturesState(readings=[SensorReading(name="CCD", value=await self._get_temperature())]),
        )

    async def set_window(self, left: int, top: int, width: int, height: int, **kwargs: Any) -> None:
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
        log.info("Setting window to %dx%d at %d,%d...", width, height, left, top)
        await self.comm.set_state(IWindow, WindowState(x=left, y=top, width=width, height=height))

    async def set_binning(self, x: int, y: int, **kwargs: Any) -> None:
        """Set the camera binning.

        Args:
            x: X binning.
            y: Y binning.

        Raises:
            ValueError: If binning could not be set.
        """
        self._binning = x
        log.info("Setting binning to %dx%d...", x, x)
        await self.comm.set_state(IBinning, BinningState(x=x, y=y))

    async def _expose(self, exposure_time: float, open_shutter: bool, abort_event: asyncio.Event) -> Image:
        """Actually do the exposure, should be implemented by derived classes.

        Args:
            exposure_time: The requested exposure time in s.
            open_shutter: Whether or not to open the shutter.
            abort_event: Event that gets triggered when exposure should be aborted.

        Returns:
            The actual image.

        Raises:
            GrabImageError: If exposure was not successful.
        """
        if self._camera is None:
            raise ValueError("No camera initialised.")
        camera = self._camera

        image_format = FORMATS[self._image_format]

        # set window, divide width/height by binning
        width = int(math.floor(self._window[2]) / self._binning)
        height = int(math.floor(self._window[3]) / self._binning)
        log.info(
            "Set window to %dx%d (binned %dx%d with %dx%d) at %d,%d.",
            self._window[2],
            self._window[3],
            width,
            height,
            self._binning,
            self._binning,
            self._window[0],
            self._window[1],
        )

        log.info(
            "Starting exposure with %s shutter for %s seconds and %s gain...",
            "open" if open_shutter else "closed",
            exposure_time,
            self._gain,
        )

        def _start() -> None:
            camera.set_roi(int(self._window[0]), int(self._window[1]), width, height, self._binning, image_format)
            # set status and exposure time in us
            camera.set_control_value(asi.ASI_EXPOSURE, int(exposure_time * 1e6))
            # set gain and offset
            camera.set_control_value(asi.ASI_GAIN, int(self._gain))
            camera.set_control_value(asi.ASI_OFFSET, int(self._gain_offset))
            camera.start_exposure()

        await self._run_blocking_or_raise(_start, lock=self._sdk_lock)
        await asyncio.sleep(0.01)

        # wait for image -- runs the whole poll loop as a single blocking call (see _run_blocking),
        # rather than polling get_exposure_status() every 10ms directly on the event loop. The SDK
        # lock is taken per get_exposure_status() call here, so the temperature poll can interleave.
        def _wait() -> Any:
            while True:
                with self._sdk_lock:
                    exposure_status = camera.get_exposure_status()
                if exposure_status != asi.ASI_EXP_WORKING or abort_event.is_set():
                    return exposure_status
                time.sleep(0.01)

        exposure_timeout = exposure_time + _EXPOSURE_WAIT_MARGIN
        status = await self._run_blocking_or_raise(_wait, timeout=exposure_timeout)

        if abort_event.is_set():
            await self._run_blocking_or_raise(camera.stop_exposure, lock=self._sdk_lock)
            await self._change_exposure_status(ExposureStatus.IDLE)
            raise InterruptedError("Aborted exposure.")
        if status != asi.ASI_EXP_SUCCESS:
            raise exc.GrabImageError(f"Could not capture image: {status}")

        log.info("Exposure finished, reading out...")
        await self._change_exposure_status(ExposureStatus.READOUT)

        def _readout() -> tuple[Any, Any]:
            buffer = camera.get_data_after_exposure()
            whbi = camera.get_roi_format()
            return buffer, whbi

        buffer, whbi = await self._run_blocking_or_raise(_readout, timeout=_READOUT_TIMEOUT, lock=self._sdk_lock)

        # decide on image format
        shape = [whbi[1], whbi[0]]
        data: NDArray[Any]
        if image_format == asi.ASI_IMG_RAW8:
            data = np.frombuffer(buffer, dtype=np.uint8)
        elif image_format == asi.ASI_IMG_RAW16:
            data = np.frombuffer(buffer, dtype=np.uint16)
        elif image_format == asi.ASI_IMG_RGB24:
            shape.append(3)
            data = np.frombuffer(buffer, dtype=np.uint8)
        else:
            raise exc.GrabImageError("Unknown image format.")

        data = data.reshape(shape)

        # special treatment for RGB images
        if image_format == asi.ASI_IMG_RGB24:
            # convert BGR to RGB
            data = data[:, :, ::-1]

            # now we need to separate the R, G, and B images
            # this is easiest done by shifting the RGB axis from last to first position
            # i.e. we go from RGBRGBRGBRGBRGB to RRRRRGGGGGBBBBB
            data = np.moveaxis(data, 2, 0)

        date_obs = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")

        image = Image(cast(NDArray[Any], data))
        image.header["DATE-OBS"] = (date_obs, "Date and time of start of exposure")
        image.header["EXPTIME"] = (exposure_time, "Exposure time [s]")

        image.header["INSTRUME"] = (self._camera_name, "Name of instrument")

        image.header["XBINNING"] = image.header["DET-BIN1"] = (self._binning, "Binning factor used on X axis")
        image.header["YBINNING"] = image.header["DET-BIN2"] = (self._binning, "Binning factor used on Y axis")

        image.header["XORGSUBF"] = (self._window[0], "Subframe origin on X axis")
        image.header["YORGSUBF"] = (self._window[1], "Subframe origin on Y axis")

        image.header["DATAMIN"] = (float(np.min(data)), "Minimum data value")
        image.header["DATAMAX"] = (float(np.max(data)), "Maximum data value")
        image.header["DATAMEAN"] = (float(np.mean(data)), "Mean data value")

        image.header["DET-PIXL"] = (self._camera_info["PixelSize"] / 1000.0, "Size of detector pixels (square) [mm]")
        image.header["DET-GAIN"] = (self._camera_info["ElecPerADU"] * self._gain, "Detector gain [e-/ADU]")
        image.header["OFFSET"] = (self._gain_offset, "Black-level offset used for exposure")

        if image_format in [asi.ASI_IMG_RAW8, asi.ASI_IMG_RAW16]:
            image.header["BAYERPAT"] = image.header["COLORTYP"] = ("GBRG", "Bayer pattern for colors")

        image.header["DET-TEMP"] = (await self._get_temperature(), "CCD temperature [C]")

        self.set_biassec_trimsec(image.header, *self._window)

        log.info("Readout finished.")
        return image

    async def set_image_format(self, fmt: ImageFormat, **kwargs: Any) -> None:
        """Set the camera image format.

        Args:
            fmt: New image format.

        Raises:
            ValueError: If format could not be set.
        """
        if fmt not in FORMATS:
            raise ValueError("Unsupported image format.")
        self._image_format = fmt
        await self.comm.set_state(IImageFormat, ImageFormatState(image_format=fmt))

    async def _get_temperature(self) -> float:
        """Gets the temperature from the camera.

        Reading is divided by 10, since ASI_TEMPERATURE returns temp * 10
        """
        camera = self._camera

        def _get() -> float:
            return camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10  # type: ignore[union-attr]

        return await self._run_blocking_or_raise(_get, lock=self._sdk_lock)

    async def set_gain(self, gain: float, **kwargs: Any) -> None:
        """Set the camera gain.

        Args:
            gain: New camera gain.

        Raises:
            ValueError: If gain could not be set.
        """
        self._gain = gain
        await self.comm.set_state(IGain, GainState(gain=self._gain, offset=self._gain_offset))

    async def set_offset(self, offset: float, **kwargs: Any) -> None:
        """Set the camera offset.

        Args:
            offset: New camera offset.

        Raises:
            ValueError: If offset could not be set.
        """
        self._gain_offset = offset
        await self.comm.set_state(IGain, GainState(gain=self._gain, offset=self._gain_offset))


class AsiCoolCamera(AsiCamera, ICooling):
    """A pyobs module for ASI cameras with cooling."""

    def __init__(self, setpoint: int = -20, **kwargs: Any):
        """Initializes a new AsiCoolCamera.

        Args:
            setpoint: Cooling temperature setpoint.
        """
        AsiCamera.__init__(self, **kwargs)
        self._temp_setpoint = setpoint

    async def open(self) -> None:
        """Open module."""
        await AsiCamera.open(self)

        if not self._camera_info["IsCoolerCam"]:
            raise ValueError("Camera has no support for cooling.")

        await self.set_cooling(True, self._temp_setpoint)

    async def _publish_temperatures(self) -> None:
        await super()._publish_temperatures()
        enabled, setpoint, power = await self._get_cooling_status()
        await self.comm.set_state(ICooling, CoolingState(setpoint=setpoint, power=int(power), enabled=enabled))

    async def add_custom_fits_headers(self, image: Image) -> None:
        """Add cooling headers on top of AsiCamera's."""
        await super().add_custom_fits_headers(image)
        _, setpoint, power = await self._get_cooling_status()
        image.header["DET-COOL"] = (int(power), "Cooler power [percent]")
        image.header["DET-TSET"] = (setpoint, "Cooler setpoint [C]")

    async def _get_cooling_status(self) -> tuple[bool, float, float]:
        if self._camera is None:
            raise ValueError("No camera initialised.")
        camera = self._camera

        def _get() -> tuple[bool, float, float]:
            enabled = camera.get_control_value(asi.ASI_COOLER_ON)[0]
            setpoint = camera.get_control_value(asi.ASI_TARGET_TEMP)[0]
            power = camera.get_control_value(asi.ASI_COOLER_POWER_PERC)[0]
            return bool(enabled), float(setpoint), float(power)

        return await self._run_blocking_or_raise(_get, lock=self._sdk_lock)

    async def set_cooling(self, enabled: bool, setpoint: float, **kwargs: Any) -> None:
        """Enables/disables cooling and sets setpoint.

        Args:
            enabled: Enable or disable cooling.
            setpoint: Setpoint in celsius for the cooling.

        Raises:
            ValueError: If cooling could not be set.
        """
        if self._camera is None:
            raise ValueError("No camera initialised.")
        camera = self._camera

        def _set() -> None:
            if enabled:
                log.info("Enabling cooling with a setpoint of %.2f°C...", setpoint)
                camera.set_control_value(asi.ASI_TARGET_TEMP, int(setpoint))
                camera.set_control_value(asi.ASI_COOLER_ON, 1)
            else:
                log.info("Disabling cooling...")
                camera.set_control_value(asi.ASI_COOLER_ON, 0)

        await self._run_blocking_or_raise(_set, lock=self._sdk_lock)

        enabled_status, setpoint_status, power = await self._get_cooling_status()
        await self.comm.set_state(
            ICooling, CoolingState(setpoint=setpoint_status, power=int(power), enabled=enabled_status)
        )


__all__ = ["AsiCamera", "AsiCoolCamera"]
