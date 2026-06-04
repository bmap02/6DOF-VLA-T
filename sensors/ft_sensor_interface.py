"""sensors/ft_sensor_interface.py — simulated 6-axis F/T sensor.

Wraps raw Webots TouchSensor readings (or simulated values) into a clean
FTReading dataclass with noise, drift, and calibration offset support.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FTReading:
    fx: float; fy: float; fz: float   # N
    tx: float; ty: float; tz: float   # N·m
    timestamp: float                   # s

    @property
    def force_magnitude(self) -> float:
        return float(np.sqrt(self.fx**2 + self.fy**2 + self.fz**2))

    @property
    def torque_magnitude(self) -> float:
        return float(np.sqrt(self.tx**2 + self.ty**2 + self.tz**2))

    def to_dict(self) -> dict:
        return {
            "fx": round(self.fx, 4), "fy": round(self.fy, 4), "fz": round(self.fz, 4),
            "tx": round(self.tx, 4), "ty": round(self.ty, 4), "tz": round(self.tz, 4),
            "force_mag":  round(self.force_magnitude, 4),
            "torque_mag": round(self.torque_magnitude, 4),
            "timestamp":  round(self.timestamp, 4),
        }


class ForceTorqueSensor:
    """
    6-axis F/T sensor model.

    Parameters
    ----------
    force_noise_std  : N    — Gaussian noise on each force axis
    torque_noise_std : N·m  — Gaussian noise on each torque axis
    drift_rate       : N/s  — slow linear bias drift per axis
    """

    def __init__(
        self,
        force_noise_std:  float = 0.05,
        torque_noise_std: float = 0.002,
        drift_rate:       float = 0.001,
        rng_seed: int | None = None,
    ) -> None:
        self.fn  = force_noise_std
        self.tn  = torque_noise_std
        self.dr  = drift_rate
        self.rng = np.random.default_rng(rng_seed)
        self._bias = np.zeros(6)
        self._zero = np.zeros(6)   # tare / zero offset

    def tare(self) -> None:
        """Zero the sensor at current reading (call before grasping)."""
        self._zero = self._bias.copy()

    def read(
        self,
        true_wrench: tuple[float, float, float, float, float, float],
        timestamp: float,
    ) -> FTReading:
        """
        Convert a true wrench into a noisy sensor reading.

        Parameters
        ----------
        true_wrench : (fx, fy, fz, tx, ty, tz) in sensor frame
        timestamp   : simulation time in seconds
        """
        raw = np.array(true_wrench, dtype=float)

        # Drift update
        self._bias += self.rng.normal(0.0, self.dr * 0.032, 6)  # per 32ms step

        # Gaussian noise
        noise = np.concatenate([
            self.rng.normal(0.0, self.fn,  3),
            self.rng.normal(0.0, self.tn, 3),
        ])

        reading = raw + self._bias - self._zero + noise

        return FTReading(
            fx=reading[0], fy=reading[1], fz=reading[2],
            tx=reading[3], ty=reading[4], tz=reading[5],
            timestamp=timestamp,
        )