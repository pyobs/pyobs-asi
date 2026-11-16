import asyncio
import sys
import traceback
from typing import Any

import numpy as np
import qasync  # type: ignore
import zwoasi as asi  # type: ignore
from astropy.io import fits
from pyobs.utils.enums import ImageFormat
from pyobs.utils.gui.camera import (
    BinningWidget,
    CoolingWidget,
    DataDisplayWidget,
    ExposeWidget,
    ExposureTimeWidget,
    GainOffsetWidget,
    ImageFormatWidget,
    ListPickerDialog,
)
from pyobs.utils.gui.camera.windowingwidget import WindowingWidget
from pyobs.utils.parallel import event_wait
from PySide6 import QtCore, QtWidgets  # type: ignore[import-untyped]

from .coolingramper import SdkCoolingRamper

# the first entry is the default in the GUI
FORMATS = {
    ImageFormat.INT16: asi.ASI_IMG_RAW16,
    ImageFormat.INT8: asi.ASI_IMG_RAW8,
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, camera: asi.Camera, camera_info: dict[str, Any]) -> None:
        super().__init__()

        self.camera = camera
        # exposure times below the SDK minimum give garbage frames
        controls = camera.get_controls()
        self.min_exposure_us = int(controls["Exposure"]["MinValue"])

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        global_layout = QtWidgets.QHBoxLayout()
        self.central_widget.setLayout(global_layout)
        self.widgets_frame = QtWidgets.QGroupBox()
        self.data_display_widget = DataDisplayWidget()
        global_layout.addWidget(self.widgets_frame)
        global_layout.addWidget(self.data_display_widget)

        layout = QtWidgets.QVBoxLayout()
        self.widgets_frame.setLayout(layout)
        self.window_widget = WindowingWidget(camera_info["MaxWidth"], camera_info["MaxHeight"])
        layout.addWidget(self.window_widget)
        bins = camera_info.get("SupportedBins", [1])
        self.binning_widget = BinningWidget([(b, b) for b in bins])
        self.binning_widget.binning_changed.connect(self.window_widget.set_binning)
        layout.addWidget(self.binning_widget)
        self.format_widget = ImageFormatWidget(list(FORMATS.keys()))
        layout.addWidget(self.format_widget)
        self.exposure_time = ExposureTimeWidget()
        layout.addWidget(self.exposure_time)

        self.gain_offset: GainOffsetWidget | None = None
        if "Gain" in controls and "Offset" in controls:
            self.gain_offset = GainOffsetWidget(
                (controls["Gain"]["MinValue"], controls["Gain"]["MaxValue"]),
                (controls["Offset"]["MinValue"], controls["Offset"]["MaxValue"]),
                camera.get_control_value(asi.ASI_GAIN)[0],
                camera.get_control_value(asi.ASI_OFFSET)[0],
            )
            layout.addWidget(self.gain_offset)

        # high speed mode gives odd edge pixels on the ASI2600MM Pro and the camera may keep the setting, so reset it
        if "HighSpeedMode" in controls:
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 0)

        self.cooling: CoolingWidget | None = None
        # Cooling is ramped in steps, so that warming up does not shock the sensor.
        self.cooling_ramper = SdkCoolingRamper(camera)
        self.cooling_target: tuple[bool, float] | None = None
        self.cooling_wakeup = asyncio.Event()
        self.cooling_task: asyncio.Task[None] | None = None
        if "CoolerOn" in controls and "TargetTemp" in controls:
            self.cooling = CoolingWidget(
                (controls["TargetTemp"]["MinValue"], controls["TargetTemp"]["MaxValue"]),
                bool(camera.get_control_value(asi.ASI_COOLER_ON)[0]),
                camera.get_control_value(asi.ASI_TARGET_TEMP)[0],
            )
            self.cooling.cooling_changed.connect(self._cooling_changed)
            layout.addWidget(self.cooling)
            self.cooling_timer = QtCore.QTimer()
            self.cooling_timer.setInterval(2000)
            self.cooling_timer.timeout.connect(self._update_cooling)
            self.cooling_timer.start()

        self.expose = ExposeWidget()
        layout.addWidget(self.expose)

        self.abort_exposure = asyncio.Event()
        self.expose.expose_clicked.connect(self._expose_clicked)
        self.expose.abort_clicked.connect(self._abort_clicked)

    @qasync.asyncSlot()  # type: ignore
    async def _cooling_changed(self, enabled: bool, set_point: float) -> None:
        # a new target replaces a running ramp, which continues from the current setpoint
        self.cooling_target = (enabled, set_point)
        self.cooling_wakeup.set()
        if self.cooling_task is None or self.cooling_task.done():
            self.cooling_task = asyncio.create_task(self._ramp_cooling())

    async def _ramp_cooling(self) -> None:
        loop = asyncio.get_running_loop()
        while self.cooling_target is not None:
            target = self.cooling_target
            self.cooling_wakeup.clear()
            try:
                delay = await loop.run_in_executor(None, self.cooling_ramper.step, *target)
            except Exception:
                traceback.print_exc()
                delay = 5.0
            if delay is None:
                if self.cooling_target is target:
                    self.cooling_target = None
                continue
            try:
                await asyncio.wait_for(self.cooling_wakeup.wait(), delay)
            except TimeoutError:
                pass

    @qasync.asyncSlot()  # type: ignore
    async def _update_cooling(self) -> None:
        if self.cooling is None:
            return

        def _read() -> tuple[int, int]:
            return (
                self.camera.get_control_value(asi.ASI_TEMPERATURE)[0],
                self.camera.get_control_value(asi.ASI_COOLER_POWER_PERC)[0],
            )

        temperature, power = await asyncio.get_running_loop().run_in_executor(None, _read)
        # the SDK reports the temperature in units of 0.1 degC
        self.cooling.set_temperature(temperature / 10.0)
        self.cooling.set_power(power)

    @qasync.asyncSlot()  # type: ignore
    async def _expose_clicked(self) -> None:
        try:
            await self._expose()
        except Exception:
            # don't leave the GUI in the exposing state
            traceback.print_exc()
        finally:
            self.abort_exposure = asyncio.Event()
            self.expose.set_exposures_left()

    async def _expose(self) -> None:
        left, top, width, height = self.window_widget.values
        # the SDK needs a width that is a multiple of 8 and a height that is a multiple of 2
        width = max(8, width // 8 * 8)
        height = max(2, height // 2 * 2)
        binning = self.window_widget.binning[0]
        image_type = FORMATS[self.format_widget.value]
        loop = asyncio.get_running_loop()

        # WindowingWidget already returns binned coordinates (its maxima are the binned
        # dimensions), and set_roi works in binned units too, so no further division is needed.
        exposure_us = max(int(self.exposure_time.value * 1e6), self.min_exposure_us)
        self.expose.start_exposure(exposure_us / 1e6)

        gain_offset = self.gain_offset

        def _start() -> None:
            if gain_offset is not None:
                self.camera.set_control_value(asi.ASI_GAIN, gain_offset.gain)
                self.camera.set_control_value(asi.ASI_OFFSET, gain_offset.offset)
            self.camera.set_roi(left, top, width, height, bins=binning, image_type=image_type)
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
            self.camera.start_exposure()

        await loop.run_in_executor(None, _start)

        def _get_status() -> int:
            return self.camera.get_exposure_status()

        while await loop.run_in_executor(None, _get_status) == asi.ASI_EXP_WORKING:
            if self.abort_exposure.is_set():
                await loop.run_in_executor(None, self.camera.stop_exposure)
                break
            await event_wait(self.abort_exposure, 0.1)

        if not self.abort_exposure.is_set():
            status = await loop.run_in_executor(None, _get_status)
            if status != asi.ASI_EXP_SUCCESS:
                print(f"Exposure failed (status {status}), discarding frame.")
                return
            buffer = await loop.run_in_executor(None, self.camera.get_data_after_exposure)
            whbi = await loop.run_in_executor(None, self.camera.get_roi_format)
            shape = [whbi[1], whbi[0]]
            dtype = np.uint8 if image_type == asi.ASI_IMG_RAW8 else np.uint16
            data = np.frombuffer(buffer, dtype=dtype).reshape(shape)
            print(f"Frame: min={data.min()} max={data.max()} mean={data.mean():.1f}")
            if data.max() == 0:
                print("Frame contains only zeros, discarding it.")
                return

            image = fits.PrimaryHDU(data)
            self.data_display_widget.set_data(image)

    @qasync.asyncSlot()  # type: ignore
    async def _abort_clicked(self) -> None:
        self.abort_exposure.set()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.cooling_target is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Cooling ramp running",
                "The cooling setpoint is still being ramped. Closing now leaves the camera at an intermediate "
                "setpoint with the cooler on. Close anyway?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.camera.close()
        super().closeEvent(event)


async def async_main(app: QtWidgets.QApplication) -> None:
    sdk_path = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/libASICamera2.so"
    loop = asyncio.get_running_loop()

    def _init() -> int:
        asi.init(sdk_path)
        return asi.get_num_cameras()

    if await loop.run_in_executor(None, _init) == 0:
        print("No devices found. Exiting...")
        return
    cameras_found = await loop.run_in_executor(None, asi.list_cameras)
    device_picker = ListPickerDialog(cameras_found)
    if device_picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        print("No device selected. Exiting...")
        return
    camera_id = device_picker.comboBox().currentIndex()

    def _open_camera() -> asi.Camera:
        camera = asi.Camera(camera_id)
        camera.disable_dark_subtract()
        return camera

    camera = await loop.run_in_executor(None, _open_camera)
    camera_info = await loop.run_in_executor(None, camera.get_camera_property)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)
    main_window = MainWindow(camera, camera_info)
    main_window.show()
    app.setQuitOnLastWindowClosed(True)
    await app_close_event.wait()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    # don't quit when the device picker closes, the main window is not shown yet
    app.setQuitOnLastWindowClosed(False)
    with qasync.QEventLoop(app) as loop:
        loop.run_until_complete(async_main(app))


if __name__ == "__main__":
    main()
