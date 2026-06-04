"""physics/material_properties.py — shared material constants.

Every entry is a dict with:
  E        : Young's modulus (Pa)       — stiffness
  nu       : Poisson's ratio            — lateral strain
  mu_s     : static friction coefficient
  mu_k     : kinetic friction coefficient
  compliance: inverse stiffness (m/N)  — convenience alias
  category : semantic tag for TouchNet
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float          # Young's modulus (Pa)
    nu: float         # Poisson's ratio
    mu_s: float       # static friction
    mu_k: float       # kinetic friction
    category: str     # 'rigid' | 'soft' | 'elastic' | 'fragile' | 'slippery'

    @property
    def compliance(self) -> float:
        return 1.0 / self.E if self.E > 0 else float("inf")


# ── Material registry ─────────────────────────────────────────────────────────

MATERIALS: dict[str, Material] = {
    # ── Rigid ────────────────────────────────────────────────────────────────
    "steel":        Material("steel",        200e9, 0.29, 0.74, 0.57, "rigid"),
    "aluminum":     Material("aluminum",      69e9, 0.33, 0.61, 0.47, "rigid"),
    "wood_pine":    Material("wood_pine",     10e9, 0.35, 0.45, 0.35, "rigid"),
    "abs_plastic":  Material("abs_plastic",   2.3e9,0.38, 0.50, 0.40, "rigid"),
    "ceramic":      Material("ceramic",      300e9, 0.22, 0.60, 0.45, "rigid"),

    # ── Soft ─────────────────────────────────────────────────────────────────
    "foam_soft":    Material("foam_soft",    50e3,  0.45, 0.55, 0.45, "soft"),
    "foam_rigid":   Material("foam_rigid",   1e6,   0.40, 0.50, 0.40, "soft"),
    "silicone":     Material("silicone",     1e6,   0.48, 0.65, 0.55, "soft"),
    "rubber":       Material("rubber",       5e6,   0.49, 0.80, 0.70, "soft"),
    "gel":          Material("gel",          10e3,  0.49, 0.30, 0.25, "soft"),

    # ── Elastic ──────────────────────────────────────────────────────────────
    "latex":        Material("latex",        1e6,   0.49, 0.75, 0.65, "elastic"),
    "neoprene":     Material("neoprene",     3e6,   0.49, 0.70, 0.60, "elastic"),
    "polyurethane": Material("polyurethane", 30e6,  0.45, 0.60, 0.50, "elastic"),

    # ── Fragile ──────────────────────────────────────────────────────────────
    "glass":        Material("glass",        70e9,  0.22, 0.40, 0.30, "fragile"),
    "eggshell":     Material("eggshell",     30e9,  0.26, 0.35, 0.28, "fragile"),
    "thin_ceramic": Material("thin_ceramic", 250e9, 0.22, 0.50, 0.40, "fragile"),
    "wafer":        Material("wafer",        130e9, 0.28, 0.20, 0.15, "fragile"),

    # ── Slippery ─────────────────────────────────────────────────────────────
    "teflon":       Material("teflon",       0.5e9, 0.46, 0.04, 0.04, "slippery"),
    "wet_glass":    Material("wet_glass",    70e9,  0.22, 0.10, 0.08, "slippery"),
    "oily_metal":   Material("oily_metal",  200e9, 0.29, 0.08, 0.05, "slippery"),
    "ice":          Material("ice",          9e9,   0.33, 0.03, 0.01, "slippery"),
    "polished_marble": Material("polished_marble", 50e9, 0.28, 0.12, 0.10, "slippery"),
}


def get(name: str) -> Material:
    """Retrieve a material by name; raises KeyError if not found."""
    return MATERIALS[name]


def by_category(category: str) -> list[Material]:
    """Return all materials matching a category tag."""
    return [m for m in MATERIALS.values() if m.category == category]