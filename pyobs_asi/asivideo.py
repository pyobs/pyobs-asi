import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

import zwoasi as asi  # type: ignore
from pyobs.images import Image
from pyobs.interfaces import (
    Binning,
    BinningCapabilities,
    BinningState,
    ExposureTimeState,
    GainState,
    IBinning,
    IExposureTime,
    IGain,
    IImageFormat,
    ImageFormatCapabilities,
    ImageFormatState,
    IWindow,
    WindowCapabilities,
    WindowState,
)
from pyobs.modules.camera import BaseVideo, Frame
from pyobs.utils.enums import ImageFormat, ImageType

from .blocking import SDK_CALL_TIMEOUT, BlockingSdkMixin

log = logging.getLogger(__name__)

# RGB24 isn't offered: the live view and the raw stream expect single-channel frames
VIDEO_FORMATS = {
    ImageFormat.INT8: asi.ASI_IMG_RAW8,
    ImageFormat.INT16: asi.ASI_IMG_RAW16,
}

# The reader thread waits for a frame in slices of this length (in ms), releasing the SDK lock in
# between, so setters never wait longer than this for the lock -- even with long exposures.
_POLL_TIMEOUT_MS = 200

# ASI_ERROR_TIMEOUT, i.e. no frame within _POLL_TIMEOUT_MS
_ASI_ERROR_TIMEOUT = 11

# frames() warns if no frame arrived for this long, on top of the exposure time
_FRAME_WAIT_MARGIN = 30.0

# frames queued between the reader thread and the event loop; latest wins beyond that
_QUEUE_SIZE = 5


class AsiVideo(BlockingSdkMixin, BaseVideo, IExposureTime, IGain, IWindow, IBinning, IImageFormat):
    """A pyobs module streaming video from ASI cameras.

    The camera is opened for the lifetime of the module and runs in video mode while the module is
    active. For long single exposures, use :class:`~pyobs_asi.AsiCamera` instead; only one of them
    can hold the camera at a time.

    The SDK gives no per-frame timestamp in video mode, so exposure start times are estimated from
    the arrival time, the exposure time and the ``readout_time`` parameter of
    :class:`~pyobs.modules.camera.BaseVideo`, which should be set for the sensor in use.
    """

    __module__ = "pyobs_asi"

    def __init__(
        self,
        camera: str,
        sdk: str = "/usr/local/lib/libASICamera2.so",
        exposure_time: float = 0.1,
        gain: float = 1.0,
        offset: float = 50.0,
        image_format: ImageFormat | str = ImageFormat.INT16,
        **kwargs: Any,
    ):
        """Initializes a new AsiVideo.

        Args:
            camera: Name of camera to use.
            sdk: Path to .so file from ASI SDK.
            exposure_time: Initial exposure time in seconds, also restored by reset().
            gain: Initial gain, also restored by reset().
            offset: Initial offset, also restored by reset().
            image_format: Initial image format (int8 or int16), also restored by reset().
        """
        BaseVideo.__init__(self, **kwargs)

        self._camera_name = camera
        self._sdk_path = sdk
        self._camera: asi.Camera | None = None
        self._camera_info: dict[str, Any] = {}

        self._default_exposure_time = exposure_time
        self._default_gain = gain
        self._default_offset = offset
        self._default_image_format = ImageFormat(image_format)
        if self._default_image_format not in VIDEO_FORMATS:
            raise ValueError(f"Unsupported image format: {image_format}")

        self._exposure_time = exposure_time
        self._gain = gain
        self._gain_offset = offset
        self._image_format = self._default_image_format
        self._window = (0, 0, 0, 0)
        self._binning = 1

        # The ASI SDK is not thread-safe: the reader thread in frames() and the setters would
        # otherwise hit the same camera handle concurrently.
        self._sdk_lock = threading.Lock()
        # generation bumps come from SDK threads (under _sdk_lock) and from the event loop
        self._generation_lock = threading.Lock()
        # set by setters waiting for _sdk_lock, so the reader thread lets them in between polls
        self._settings_pending = threading.Event()
        # whether frames() has video capture running, so _apply_settings() knows to restart it
        self._capturing = False

    async def open(self) -> None:
        """Open module."""

        def _open() -> tuple[asi.Camera, dict[str, Any]]:
            asi.init(self._sdk_path)

            if asi.get_num_cameras() == 0:
                raise ValueError("No cameras found")

            # index() raises ValueError, if camera could not be found
            camera = asi.Camera(asi.list_cameras().index(self._camera_name))
            camera_info = camera.get_camera_property()

            camera.disable_dark_subtract()
            camera.set_control_value(asi.ASI_WB_B, 99)
            camera.set_control_value(asi.ASI_WB_R, 75)
            camera.set_control_value(asi.ASI_GAMMA, 50)
            camera.set_control_value(asi.ASI_FLIP, 0)
            camera.stop_exposure()
            camera.stop_video_capture()
            return camera, camera_info

        self._camera, self._camera_info = await self._run_blocking_or_raise(_open, lock=self._sdk_lock)
        self._window = (0, 0, self._camera_info["MaxWidth"], self._camera_info["MaxHeight"])
        await self._apply_settings()

        log.info("Camera info:")
        for key, val in self._camera_info.items():
            log.info("  - %s: %s", key, val)

        await BaseVideo.open(self)

        # publish capabilities
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
        await self.comm.set_capabilities(
            IImageFormat, ImageFormatCapabilities(image_formats=list(VIDEO_FORMATS.keys()))
        )

        # publish initial states
        await self._publish_states()

        # start streaming
        await self.activate_camera()

    async def close(self) -> None:
        """Close the module."""
        await BaseVideo.close(self)

        camera, self._camera = self._camera, None
        if camera is not None:
            if not await self._run_blocking(camera.close, lock=self._sdk_lock):
                log.error("Timed out closing camera after %.1fs.", SDK_CALL_TIMEOUT)

    async def _publish_states(self) -> None:
        await self.comm.set_state(IExposureTime, ExposureTimeState(exposure_time=self._exposure_time))
        await self.comm.set_state(IGain, GainState(gain=self._gain, offset=self._gain_offset))
        await self.comm.set_state(IWindow, WindowState(*self._window))
        await self.comm.set_state(IBinning, BinningState(x=self._binning, y=self._binning))
        await self.comm.set_state(IImageFormat, ImageFormatState(image_format=self._image_format))

    def _new_generation(self) -> int:
        """Start a new settings generation, thread-safe (called from SDK threads, too)."""
        with self._generation_lock:
            return super()._new_generation()

    def _configure(self, camera: asi.Camera) -> None:
        """Write the current settings to the camera. Called under the SDK lock, with video stopped.

        The window is given in unbinned pixels. The SDK wants the ROI in binned pixels, with a width
        that's a multiple of 8 and a height that's a multiple of 2, so it is shrunk to fit.
        """
        bins = self._binning
        width = self._window[2] // bins // 8 * 8
        height = self._window[3] // bins // 2 * 2
        camera.set_roi(
            self._window[0] // bins,
            self._window[1] // bins,
            width,
            height,
            bins,
            VIDEO_FORMATS[self._image_format],
        )
        camera.set_control_value(asi.ASI_EXPOSURE, int(self._exposure_time * 1e6))
        camera.set_control_value(asi.ASI_GAIN, int(self._gain))
        camera.set_control_value(asi.ASI_OFFSET, int(self._gain_offset))

        # the camera may round the exposure time
        self._exposure_time = camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1e6
        self._new_generation()

    async def _apply_settings(self) -> None:
        """Write the current settings to the camera and start a new settings generation.

        Restarts a running video capture around it, which also drops the frames still queued in the
        SDK, since those were exposed with the old settings. Does nothing before open(), which
        applies the settings itself.

        Raises:
            TimeoutError: If the SDK didn't respond in time.
        """

        def _apply() -> None:
            self._settings_pending.set()
            try:
                with self._sdk_lock:
                    camera = self._camera
                    if camera is None:
                        return
                    if self._capturing:
                        camera.stop_video_capture()
                    try:
                        self._configure(camera)
                    finally:
                        if self._capturing:
                            camera.start_video_capture()
            finally:
                self._settings_pending.clear()

        await self._run_blocking_or_raise(_apply)

    async def frames(self) -> AsyncGenerator[Frame, None]:
        """Start video capture and yield frames until BaseVideo closes the iterator.

        A single reader thread polls the SDK for the whole activation and hands frames to the event
        loop. Each frame is stamped with the settings generation and exposure time in effect when
        it was read, under the same lock that _apply_settings() holds while it restarts the capture,
        so the stamp is always right.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Frame | BaseException] = asyncio.Queue(maxsize=_QUEUE_SIZE)
        stop = threading.Event()

        def _start() -> None:
            if self._camera is None:
                raise RuntimeError("Camera not connected.")
            self._camera.start_video_capture()
            self._capturing = True

        def _put(item: Frame | BaseException) -> None:
            # latest wins: a stalled event loop mustn't grow memory
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(item)

        def _read() -> None:
            while not stop.is_set():
                # let a waiting setter take the lock first
                while self._settings_pending.is_set() and not stop.is_set():
                    time.sleep(0.005)

                item: Frame | BaseException | None = None
                with self._sdk_lock:
                    camera = self._camera
                    if camera is None or not self._capturing:
                        return
                    try:
                        data = camera.capture_video_frame(timeout=_POLL_TIMEOUT_MS)
                        item = Frame(data=data, exposure_time=self._exposure_time, generation=self.generation)
                    except asi.ZWO_IOError as e:
                        if getattr(e, "error_code", None) != _ASI_ERROR_TIMEOUT:
                            item = e
                    except Exception as e:
                        item = e

                if item is not None:
                    try:
                        loop.call_soon_threadsafe(_put, item)
                    except RuntimeError:
                        # event loop closed (shutdown)
                        return
                    if isinstance(item, BaseException):
                        return

        def _stop() -> None:
            self._capturing = False
            if self._camera is not None:
                self._camera.stop_video_capture()

        await self._run_blocking_or_raise(_start, lock=self._sdk_lock)
        threading.Thread(target=_read, daemon=True).start()
        try:
            while True:
                timeout = self._exposure_time + _FRAME_WAIT_MARGIN
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except TimeoutError:
                    log.warning("No frame from camera for %.1fs.", timeout)
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # don't join the reader: if it hangs in the SDK, it must not block deactivation
            stop.set()
            try:
                if not await self._run_blocking(_stop, lock=self._sdk_lock):
                    log.error("Timed out stopping video capture after %.1fs.", SDK_CALL_TIMEOUT)
            except Exception:
                log.exception("Error stopping video capture.")

    async def _finish_image(self, image: Image, broadcast: bool, image_type: ImageType) -> tuple[Image, str]:
        """Add camera identity and settings headers, then finish up as usual."""
        image.header["INSTRUME"] = (self._camera_name, "Name of instrument")
        image.header["XBINNING"] = image.header["DET-BIN1"] = (self._binning, "Binning factor used on X axis")
        image.header["YBINNING"] = image.header["DET-BIN2"] = (self._binning, "Binning factor used on Y axis")
        image.header["XORGSUBF"] = (self._window[0], "Subframe origin on X axis")
        image.header["YORGSUBF"] = (self._window[1], "Subframe origin on Y axis")
        image.header["GAIN"] = (self._gain, "Gain used for exposure")
        image.header["OFFSET"] = (self._gain_offset, "Black-level offset used for exposure")
        if "PixelSize" in self._camera_info:
            image.header["DET-PIXL"] = (self._camera_info["PixelSize"] / 1000.0, "Size of detector pixels [mm]")
        if "ElecPerADU" in self._camera_info:
            image.header["DET-GAIN"] = (self._camera_info["ElecPerADU"] * self._gain, "Detector gain [e-/ADU]")
        if self._camera_info.get("IsColorCam"):
            image.header["BAYERPAT"] = image.header["COLORTYP"] = ("GBRG", "Bayer pattern for colors")
        return await super()._finish_image(image, broadcast, image_type)

    async def _set(self, update: Callable[[], None]) -> None:
        """Change settings via update(), write them to the camera and publish the new states.

        If the camera rejects the new settings, the old ones are restored.
        """
        old = self._settings()
        update()
        try:
            await self._apply_settings()
        except Exception:
            self._restore_settings(old)
            await self._apply_settings()
            raise
        finally:
            await self._publish_states()

    def _settings(self) -> tuple[Any, ...]:
        return self._exposure_time, self._gain, self._gain_offset, self._window, self._binning, self._image_format

    def _restore_settings(self, settings: tuple[Any, ...]) -> None:
        (
            self._exposure_time,
            self._gain,
            self._gain_offset,
            self._window,
            self._binning,
            self._image_format,
        ) = settings

    async def set_exposure_time(self, exposure_time: float, **kwargs: Any) -> None:
        """Set the exposure time in seconds.

        Args:
            exposure_time: Exposure time in seconds.

        Raises:
            ValueError: If exposure time is negative.
        """
        if exposure_time < 0:
            raise ValueError("Exposure time must not be negative.")
        log.info("Setting exposure time to %.3fs...", exposure_time)
        await self._set(lambda: setattr(self, "_exposure_time", exposure_time))

    async def set_gain(self, gain: float, **kwargs: Any) -> None:
        """Set the camera gain.

        Args:
            gain: New camera gain.
        """
        log.info("Setting gain to %s...", gain)
        await self._set(lambda: setattr(self, "_gain", gain))

    async def set_offset(self, offset: float, **kwargs: Any) -> None:
        """Set the camera offset.

        Args:
            offset: New camera offset.
        """
        log.info("Setting offset to %s...", offset)
        await self._set(lambda: setattr(self, "_gain_offset", offset))

    async def set_window(self, left: int, top: int, width: int, height: int, **kwargs: Any) -> None:
        """Set the camera window, in unbinned pixels.

        Args:
            left: X offset of window.
            top: Y offset of window.
            width: Width of window.
            height: Height of window.
        """
        log.info("Setting window to %dx%d at %d,%d...", width, height, left, top)
        await self._set(lambda: setattr(self, "_window", (left, top, width, height)))

    async def set_binning(self, x: int, y: int, **kwargs: Any) -> None:
        """Set the camera binning.

        Args:
            x: X binning.
            y: Y binning, must equal x.

        Raises:
            ValueError: If x and y differ.
        """
        if x != y:
            raise ValueError("ASI cameras only support square binning.")
        log.info("Setting binning to %dx%d...", x, y)
        await self._set(lambda: setattr(self, "_binning", x))

    async def set_image_format(self, fmt: ImageFormat, **kwargs: Any) -> None:
        """Set the camera image format.

        Args:
            fmt: New image format.

        Raises:
            ValueError: If format is not supported.
        """
        if fmt not in VIDEO_FORMATS:
            raise ValueError("Unsupported image format.")
        log.info("Setting image format to %s...", fmt)
        await self._set(lambda: setattr(self, "_image_format", fmt))

    async def reset(self, **kwargs: Any) -> None:
        """Reset image type, data pipeline, exposure time, gain, offset and image format to their defaults."""
        await BaseVideo.reset(self, **kwargs)

        def _update() -> None:
            self._exposure_time = self._default_exposure_time
            self._gain = self._default_gain
            self._gain_offset = self._default_offset
            self._image_format = self._default_image_format

        await self._set(_update)


__all__ = ["AsiVideo"]
