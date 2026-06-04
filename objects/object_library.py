"""objects/object_library.py — configurable object registry.

Each GraspObject carries geometry, a Material, and a max_safe_force
threshold. Objects can be instantiated with overridden material properties
for ablation studies (e.g. vary friction only).
"""

from dataclasses import dataclass, field
from physics.material_properties import Material, MATERIALS, get as get_material


# ── Object definition ─────────────────────────────────────────────────────────

@dataclass
class GraspObject:
    name:           str
    material_name:  str
    shape:          str      # 'sphere' | 'box' | 'cylinder' | 'irregular'
    mass:           float    # kg
    size:           tuple    # (radius,) or (lx, ly, lz) in metres
    max_safe_force: float    # N — force above which object is damaged/dropped
    notes:          str = ""

    # Override individual material props (e.g. for ablation)
    _material_override: Material | None = field(default=None, repr=False)

    @property
    def material(self) -> Material:
        if self._material_override is not None:
            return self._material_override
        return get_material(self.material_name)

    def with_friction(self, mu_s: float, mu_k: float) -> "GraspObject":
        """Return a copy with friction overridden — useful for slippery variants."""
        import dataclasses
        base = self.material
        overridden = Material(
            name     = base.name + f"_mu{mu_s:.2f}",
            E        = base.E,
            nu       = base.nu,
            mu_s     = mu_s,
            mu_k     = mu_k,
            category = base.category,
        )
        return dataclasses.replace(self, _material_override=overridden)

    def with_stiffness(self, E: float) -> "GraspObject":
        """Return a copy with Young's modulus overridden."""
        import dataclasses
        base = self.material
        overridden = Material(
            name     = base.name + f"_E{E:.0e}",
            E        = E,
            nu       = base.nu,
            mu_s     = base.mu_s,
            mu_k     = base.mu_k,
            category = base.category,
        )
        return dataclasses.replace(self, _material_override=overridden)

    def to_dict(self) -> dict:
        m = self.material
        return {
            "name":           self.name,
            "material":       m.name,
            "category":       m.category,
            "shape":          self.shape,
            "mass_kg":        self.mass,
            "size_m":         list(self.size),
            "E_Pa":           m.E,
            "mu_s":           m.mu_s,
            "mu_k":           m.mu_k,
            "compliance":     m.compliance,
            "max_safe_force": self.max_safe_force,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

OBJECTS: dict[str, GraspObject] = {

    # ── Rigid everyday objects ────────────────────────────────────────────────
    "steel_bolt":       GraspObject("steel_bolt",       "steel",       "cylinder", 0.05,  (0.008, 0.04), 500.0),
    "aluminum_can":     GraspObject("aluminum_can",     "aluminum",    "cylinder", 0.015, (0.033, 0.115),  80.0),
    "abs_block":        GraspObject("abs_block",        "abs_plastic", "box",      0.10,  (0.05, 0.05, 0.05), 200.0),
    "ceramic_mug":      GraspObject("ceramic_mug",      "ceramic",     "cylinder", 0.30,  (0.04, 0.10),   60.0),
    "wood_block":       GraspObject("wood_block",       "wood_pine",   "box",      0.20,  (0.06, 0.04, 0.03), 300.0),

    # ── Soft / deformable ─────────────────────────────────────────────────────
    "foam_cube":        GraspObject("foam_cube",        "foam_soft",   "box",      0.02,  (0.05, 0.05, 0.05),  10.0),
    "silicone_ball":    GraspObject("silicone_ball",    "silicone",    "sphere",   0.08,  (0.04,),             40.0),
    "rubber_cylinder":  GraspObject("rubber_cylinder",  "rubber",      "cylinder", 0.12,  (0.03, 0.08),        60.0),
    "gel_pad":          GraspObject("gel_pad",          "gel",         "box",      0.05,  (0.06, 0.06, 0.01),   5.0),
    "foam_ball":        GraspObject("foam_ball",        "foam_rigid",  "sphere",   0.03,  (0.035,),            15.0),

    # ── Elastic / compliant ───────────────────────────────────────────────────
    "latex_glove_roll": GraspObject("latex_glove_roll", "latex",       "cylinder", 0.04,  (0.02, 0.10),        20.0),
    "neoprene_grip":    GraspObject("neoprene_grip",    "neoprene",    "box",      0.06,  (0.04, 0.03, 0.02),  40.0),
    "pu_gasket":        GraspObject("pu_gasket",        "polyurethane","cylinder", 0.03,  (0.025, 0.015),      30.0),

    # ── Fragile ───────────────────────────────────────────────────────────────
    "glass_vial":       GraspObject("glass_vial",       "glass",       "cylinder", 0.02,  (0.012, 0.08),        8.0,
                                    notes="Failure at ~8 N radial force"),
    "egg":              GraspObject("egg",              "eggshell",    "sphere",   0.06,  (0.027,),             4.5,
                                    notes="Crush force ~4-5 N"),
    "ceramic_cup":      GraspObject("ceramic_cup",      "thin_ceramic","cylinder", 0.15,  (0.038, 0.09),       25.0),
    "silicon_wafer":    GraspObject("silicon_wafer",    "wafer",       "box",      0.01,  (0.075, 0.075, 0.001), 3.0,
                                    notes="Extremely thin; edge contact only"),

    # ── Slippery ─────────────────────────────────────────────────────────────
    "teflon_rod":       GraspObject("teflon_rod",       "teflon",      "cylinder", 0.08,  (0.015, 0.12),       80.0),
    "wet_bottle":       GraspObject("wet_bottle",       "wet_glass",   "cylinder", 0.35,  (0.035, 0.22),       60.0),
    "oily_pipe":        GraspObject("oily_pipe",        "oily_metal",  "cylinder", 0.50,  (0.025, 0.15),      200.0),
    "ice_cube":         GraspObject("ice_cube",         "ice",         "box",      0.05,  (0.03, 0.03, 0.03),  15.0),
    "marble_sphere":    GraspObject("marble_sphere",    "polished_marble","sphere",0.20,  (0.025,),            60.0),
}


def get(name: str) -> GraspObject:
    return OBJECTS[name]


def by_category(category: str) -> list[GraspObject]:
    return [o for o in OBJECTS.values() if o.material.category == category]


def all_names() -> list[str]:
    return list(OBJECTS.keys())