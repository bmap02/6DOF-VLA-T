"""physics/contact_model.py — Hertz contact + KKT constraint solver.

Produces physics-consistent contact signals (force, deformation, pressure
distribution) that feed directly into the TouchNet / CPIM input format.

References
----------
- Hertz contact theory: Johnson, Contact Mechanics, Cambridge UP, 1985
- KKT (complementarity): Stewart & Trinkle, 1996
"""

import math
from dataclasses import dataclass, field
from physics.material_properties import Material


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class ContactState:
    """Single timestep contact result passed to sensors and logger."""
    in_contact:       bool
    normal_force:     float   # N — along contact normal
    tangential_force: float   # N — magnitude of friction force
    deformation:      float   # m — Hertz indentation depth
    contact_radius:   float   # m — Hertz contact patch radius
    pressure_peak:    float   # Pa — max Hertz pressure at patch center
    slip_velocity:    float   # m/s — relative tangential velocity
    is_slipping:      bool    # KKT complementarity result
    grip_margin:      float   # [0,1] — 0 = at slip threshold, 1 = safe

    def to_dict(self) -> dict:
        return {
            "in_contact":       int(self.in_contact),
            "normal_force":     round(self.normal_force, 6),
            "tangential_force": round(self.tangential_force, 6),
            "deformation":      round(self.deformation, 8),
            "contact_radius":   round(self.contact_radius, 8),
            "pressure_peak":    round(self.pressure_peak, 2),
            "slip_velocity":    round(self.slip_velocity, 6),
            "is_slipping":      int(self.is_slipping),
            "grip_margin":      round(self.grip_margin, 4),
        }


@dataclass
class ContactNoState:
    """Returned when no contact is detected."""
    in_contact: bool = False
    normal_force: float = 0.0
    tangential_force: float = 0.0
    deformation: float = 0.0
    contact_radius: float = 0.0
    pressure_peak: float = 0.0
    slip_velocity: float = 0.0
    is_slipping: bool = False
    grip_margin: float = 1.0

    def to_dict(self) -> dict:
        return ContactState(**self.__dict__).to_dict()


# ── Hertz contact model ───────────────────────────────────────────────────────

class HertzContact:
    """
    Sphere-on-flat Hertz contact model.

    Parameters
    ----------
    finger_radius : m — equivalent fingertip / sensor radius
    mat_finger    : Material of the gripper fingertip
    """

    def __init__(self, finger_radius: float = 0.01, mat_finger: Material = None):
        from physics.material_properties import MATERIALS
        self.R = finger_radius
        self.mat_f = mat_finger or MATERIALS["silicone"]

    def _reduced_modulus(self, mat_obj: Material) -> float:
        """Plane-strain combined modulus E* (Pa)."""
        E1, nu1 = self.mat_f.E, self.mat_f.nu
        E2, nu2 = mat_obj.E,    mat_obj.nu
        return 1.0 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)

    def compute(
        self,
        normal_force: float,
        tangential_force: float,
        slip_velocity: float,
        mat_obj: Material,
    ) -> ContactState:
        """
        Compute Hertz contact geometry + KKT slip condition.

        Parameters
        ----------
        normal_force     : N  — applied normal force (>0 = compressive)
        tangential_force : N  — applied tangential force magnitude
        slip_velocity    : m/s — relative tangential velocity at contact
        mat_obj          : material of the contacted object
        """
        if normal_force <= 0.0:
            return ContactNoState()

        E_star = self._reduced_modulus(mat_obj)

        # Hertz indentation depth δ
        # δ = (9F²/16 R E*²)^(1/3)
        delta = ((9.0 * normal_force**2) / (16.0 * self.R * E_star**2)) ** (1.0 / 3.0)

        # Contact patch radius a
        a = math.sqrt(self.R * delta)

        # Peak Hertz pressure p0 = 3F / (2πa²)
        p0 = (3.0 * normal_force) / (2.0 * math.pi * a**2) if a > 0 else 0.0

        # ── KKT complementarity slip condition ──────────────────────────────
        # Coulomb friction cone: |f_t| ≤ μ_s * f_n  → no slip
        # KKT: slip_velocity * (|f_t| - μ_k * f_n) = 0
        mu_s  = mat_obj.mu_s
        mu_k  = mat_obj.mu_k
        f_max = mu_s * normal_force

        if abs(tangential_force) >= f_max or abs(slip_velocity) > 1e-5:
            is_slipping = True
            # Clamp to kinetic friction
            tangential_force = mu_k * normal_force * (
                1.0 if tangential_force >= 0 else -1.0
            )
        else:
            is_slipping = False

        grip_margin = max(0.0, 1.0 - abs(tangential_force) / (f_max + 1e-12))

        return ContactState(
            in_contact       = True,
            normal_force     = normal_force,
            tangential_force = abs(tangential_force),
            deformation      = delta,
            contact_radius   = a,
            pressure_peak    = p0,
            slip_velocity    = slip_velocity,
            is_slipping      = is_slipping,
            grip_margin      = grip_margin,
        )