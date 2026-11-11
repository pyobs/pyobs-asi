from __future__ import annotations

from typing import Any

import zwoasi as asi  # type: ignore
from pyobs.utils.coolingramp import CoolingRamp


class SdkCoolingRamper:
    """Ramps the cooling setpoint of an ASI camera that is used directly through the SDK (no pyobs module).

    All methods block on SDK calls, so call step() from a worker thread. It uses the same planner as the
    CoolingRampMixin that the AsiCoolCamera module uses.
    """

    def __init__(self, camera: Any, ramp: CoolingRamp | None = None):
        """Creates a new ramper.

        Args:
            camera: A zwoasi camera.
            ramp: Planner to use. Defaults to 1 degC/min when warming, 2 degC/min when cooling, whole degrees.
        """
        self.camera = camera
        self.ramp = ramp if ramp is not None else CoolingRamp(resolution=1.0)

    def step(self, enabled: bool, set_point: float) -> float | None:
        """Performs one step towards the target.

        Args:
            enabled: Whether the cooler should be on at the end.
            set_point: Target setpoint if enabled.

        Returns:
            Seconds to wait before the next step, or None if the target was reached.
        """
        cooler_on = bool(self.camera.get_control_value(asi.ASI_COOLER_ON)[0])
        temperature = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0  # SDK unit is 0.1 degC

        if cooler_on:
            current = float(self.camera.get_control_value(asi.ASI_TARGET_TEMP)[0])
        elif not enabled:
            return None  # already off, nothing to warm up
        else:
            # cooler is off: start at the sensor temperature, so the setpoint does not jump
            current = self.ramp.round(temperature)
            self._write(True, current)

        step = self.ramp.next_step(current, enabled, set_point, temperature)
        if step.write:
            self._write(step.enabled, step.setpoint)
        return None if step.done else step.delay

    def _write(self, enabled: bool, setpoint: float | None) -> None:
        if enabled and setpoint is not None:
            self.camera.set_control_value(asi.ASI_TARGET_TEMP, int(setpoint))
            self.camera.set_control_value(asi.ASI_COOLER_ON, 1)
        else:
            self.camera.set_control_value(asi.ASI_COOLER_ON, 0)
