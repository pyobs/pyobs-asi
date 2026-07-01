import asyncio
import sys

import numpy as np
import qasync  # type: ignore
import zwoasi as asi  # type: ignore
from astropy.io import fits
from pyobs.utils.enums import ImageFormat
from pyobs.utils.gui.camera import (
    BinningWidget,
    DataDisplayWidget,
    ExposeWidget,
    ExposureTimeWidget,
    ImageFormatWidget,
    ListPickerDialog,
)
from pyobs.utils.gui.camera.windowingwidget import WindowingWidget
from pyobs.utils.parallel import event_wait
from PySide6 import QtWidgets  # type: ignore[import-untyped]

FORMATS = {
    ImageFormat.INT8: asi.ASI_IMG_RAW8,
    ImageFormat.INT16: asi.ASI_IMG_RAW16,
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, camera_id: int) -> None:
        super().__init__()

        self.camera = asi.Camera(camera_id)
        self.camera.disable_dark_subtract()
        camera_info = self.camera.get_camera_property()

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
        self.expose = ExposeWidget()
        layout.addWidget(self.expose)

        self.abort_exposure = asyncio.Event()
        self.expose.expose_clicked.connect(self._expose_clicked)
        self.expose.abort_clicked.connect(self._abort_clicked)

    @qasync.asyncSlot()  # type: ignore
    async def _expose_clicked(self) -> None:
        left, top, width, height = self.window_widget.values
        binning = self.window_widget.binning[0]
        image_type = FORMATS[self.format_widget.value]

        self.camera.set_roi(left, top, width, height, bins=binning, image_type=image_type)
        self.camera.set_control_value(asi.ASI_EXPOSURE, int(self.exposure_time.value * 1e6))

        self.expose.start_exposure(self.exposure_time.value)
        self.camera.start_exposure()

        while self.camera.get_exposure_status() == asi.ASI_EXP_WORKING:
            if self.abort_exposure.is_set():
                self.camera.stop_exposure()
                break
            await event_wait(self.abort_exposure, 0.1)

        if not self.abort_exposure.is_set():
            loop = asyncio.get_running_loop()
            buffer = await loop.run_in_executor(None, self.camera.get_data_after_exposure)
            whbi = self.camera.get_roi_format()
            shape = [whbi[1], whbi[0]]
            dtype = np.uint8 if image_type == asi.ASI_IMG_RAW8 else np.uint16
            data = np.frombuffer(buffer, dtype=dtype).reshape(shape)

            image = fits.PrimaryHDU(data)
            self.data_display_widget.set_data(image)

        self.abort_exposure = asyncio.Event()
        self.expose.set_exposures_left()

    @qasync.asyncSlot()  # type: ignore
    async def _abort_clicked(self) -> None:
        self.abort_exposure.set()


async def async_main(app: QtWidgets.QApplication) -> None:
    sdk_path = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/libASICamera2.so"
    asi.init(sdk_path)

    if asi.get_num_cameras() == 0:
        print("No devices found. Exiting...")
        return
    cameras_found = asi.list_cameras()
    device_picker = ListPickerDialog(cameras_found)
    if device_picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        print("No device selected. Exiting...")
        return
    camera_id = device_picker.comboBox().currentIndex()

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)
    main_window = MainWindow(camera_id)
    main_window.show()
    await app_close_event.wait()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    with qasync.QEventLoop(app) as loop:
        loop.run_until_complete(async_main(app))


if __name__ == "__main__":
    main()
