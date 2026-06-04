"""sensors/tactile_array_sensor.py — simulated GelSight / DIGIT tactile array.

Models a 16×16 taxel grid. Each taxel outputs a pressure value derived from
the Hertz contact model. Output format is compatible with TouchNet input.
"""

import math
import numpy as np
from dataclasses import dataclass
from physics.contact_model import ContactState, ContactNoState


TAXEL_ROWS = 16
TAXEL_COLS = 16
SENSOR_WIDTH  = 0.020  # m — 20 mm sensor face width
SENSOR_HEIGHT = 0.020  # m — 20 mm sensor face height


@dataclass
class TactileReading:
    pressure_map: np.ndarray   # (16, 16) float32 — Pa per taxel
    total_force:  float        # N — integrated normal force
    contact_area: float        # m² — non-zero taxel area
    centroid_x:   float        # m — contact centroid x (from sensor centre)
    centroid_y:   float        # m — contact centroid y
    is_contact:   bool

    def to_dict(self) -> dict:
        flat = self.pressure_map.flatten().tolist()
        return {
            "taxel_pressures": [round(p, 2) for p in flat],
            "total_force":     round(self.total_force, 6),
            "contact_area":    round(self.contact_area, 8),
            "centroid_x":      round(self.centroid_x, 6),
            "centroid_y":      round(self.centroid_y, 6),
            "is_contact":      int(self.is_contact),
        }


class TactileSensor:
    """
    Simulated GelSight-style tactile sensor.

    Usage
    -----
    sensor = TactileSensor(noise_std=50.0)
    reading = sensor.compute(contact_state, contact_x=0.002, contact_y=-0.001)
    """

    def __init__(
        self,
        rows: int = TAXEL_ROWS,
        cols: int = TAXEL_COLS,
        width: float = SENSOR_WIDTH,
        height: float = SENSOR_HEIGHT,
        noise_std: float = 50.0,   # Pa — Gaussian taxel noise
        rng_seed: int | None = None,
    ) -> None:
        self.rows    = rows
        self.cols    = cols
        self.width   = width
        self.height  = height
        self.dx      = width  / cols   # taxel pitch x
        self.dy      = height / rows   # taxel pitch y
        self.noise   = noise_std
        self.rng     = np.random.default_rng(rng_seed)

        # Pre-compute taxel centre coordinates (sensor frame, origin = sensor centre)
        xs = np.linspace(-width / 2 + self.dx / 2, width / 2 - self.dx / 2, cols)
        ys = np.linspace(-height / 2 + self.dy / 2, height / 2 - self.dy / 2, rows)
        self.taxel_x, self.taxel_y = np.meshgrid(xs, ys)   # (rows, cols)

    def compute(
        self,
        contact: ContactState | ContactNoState,
        contact_x: float = 0.0,
        contact_y: float = 0.0,
    ) -> TactileReading:
        """
        Generate a full taxel pressure map from a ContactState.

        Parameters
        ----------
        contact   : output of HertzContact.compute()
        contact_x : m — x offset of contact centre in sensor frame
        contact_y : m — y offset of contact centre in sensor frame
        """
        if not contact.in_contact or contact.contact_radius <= 0:
            empty = np.zeros((self.rows, self.cols), dtype=np.float32)
            return TactileReading(empty, 0.0, 0.0, 0.0, 0.0, False)

        # Hertz pressure distribution: p(r) = p0 * sqrt(1 - (r/a)²)
        a  = contact.contact_radius
        p0 = contact.pressure_peak

        dx = self.taxel_x - contact_x
        dy = self.taxel_y - contact_y
        r  = np.sqrt(dx**2 + dy**2)

        inside = r < a
        pressure = np.where(inside, p0 * np.sqrt(np.maximum(0.0, 1.0 - (r / a)**2)), 0.0)

        # Add sensor noise
        pressure += self.rng.normal(0.0, self.noise, pressure.shape)
        pressure  = np.maximum(pressure, 0.0).astype(np.float32)

        # Derived scalars
        taxel_area  = self.dx * self.dy
        total_force = float(pressure.sum()) * taxel_area
        contact_area = float(inside.sum()) * taxel_area

        # Centroid (intensity-weighted)
        total_p = float(pressure.sum()) + 1e-12
        cx = float((pressure * self.taxel_x).sum()) / total_p
        cy = float((pressure * self.taxel_y).sum()) / total_p

        return TactileReading(
            pressure_map = pressure,
            total_force  = total_force,
            contact_area = contact_area,
            centroid_x   = cx,
            centroid_y   = cy,
            is_contact   = True,
        )