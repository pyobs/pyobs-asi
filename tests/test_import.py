"""Smoke tests: import the drivers and instantiate them without hardware, asserting
the interfaces they claim.

zwoasi imports cleanly with no SDK present (it logs "ASI SDK library not found" and
degrades), so no camera is needed at import/instantiate time.
"""

from pyobs.interfaces import IBinning, ICooling, IGain, IImageFormat, ITemperatures, IWindow
from pyobs.modules import Module

from pyobs_asi import AsiCamera, AsiCoolCamera


def test_instantiate_asicamera() -> None:
    camera = AsiCamera(camera="ZWO ASI1600MM Pro")
    assert isinstance(camera, Module)
    assert isinstance(camera, IWindow)
    assert isinstance(camera, IBinning)
    assert isinstance(camera, IImageFormat)
    assert isinstance(camera, IGain)
    assert isinstance(camera, ITemperatures)


def test_instantiate_asicoolcamera() -> None:
    camera = AsiCoolCamera(camera="ZWO ASI1600MM Pro")
    assert isinstance(camera, ICooling)
    assert isinstance(camera, ITemperatures)
