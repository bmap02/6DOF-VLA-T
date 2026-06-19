# 6DOF-VLA-T
**UR10e Tactile Manipulation Testbed for TouchNet / CPIM Algorithm Evaluation**

Webots R2025a · Python 3.10+ · UR10e + Robotiq 3F Gripper

---

## Overview

This repository is a simulation testbed built to run and evaluate the **TouchNet** tactile manipulation algorithm. TouchNet helps robots adapt grip pressure using early contact signals — force, deformation, stiffness, friction, and compliance.

The intern scope is **not** to build the algorithm, but to build the simulation environment that allows it to be tested safely and repeatedly. This includes the robot and gripper setup, object library, sensor models, physics engine, and benchmark scripts.

A **VLA (Vision-Language-Action) model** will be integrated in a future phase to control gripper closing force based on tactile and joint state inputs.

---

## Project Structure

```
6DOF-VLA-T/
│
├── controllers/
│   ├── ur10e_touchnet/
│   │   ├── ur10e_touchnet.py            # Main Webots robot controller (entry point)
│   │   └── runtime.ini                  # Webots Python path config
│   │
│   └── overhead_camera_detection/
│       ├── overhead_camera_detection.py # Fixed overhead camera: color detection + world coords
│       └── runtime.ini
│
├── sensors/
│   ├── joint_state_monitor.py      # Per-joint statistics: pos, vel, peak, avg, cumul, range
│   ├── tactile_array_sensor.py     # Simulated GelSight/DIGIT 16x16 tactile array
│   └── ft_sensor_interface.py      # Simulated 6-axis force/torque sensor with noise + drift
│
├── physics/
│   ├── contact_model.py            # Hertz contact model + KKT complementarity slip condition
│   └── material_properties.py      # Material registry: E, nu, mu_s, mu_k for 21 materials
│
├── objects/
│   └── object_library.py           # 22 graspable objects across 5 material categories
│
├── pipeline/
│   ├── episode_runner.py           # Episode lifecycle + VLA interface stub + state machine
│   └── log_export.py               # CPIM/TouchNet format CSV + JSON trajectory logger
│
├── benchmarks/
│   ├── benchmark_stable_grasp.py   # Benchmark: stable grasping (29/30 passing)
│   ├── benchmark_slip_detection.py # Benchmark: slip detection (TODO)
│   ├── benchmark_force_overshoot.py# Benchmark: fragile object force overshoot (TODO)
│   └── benchmark_generalization.py # Benchmark: unseen object generalization (TODO)
│
├── utils/
│   └── kinematics_utils.py         # Shared math helpers: clamp, finite diff, rolling stats
│
├── worlds/
│   └── ur10e_touchnet.wbt          # Webots world: UR10e + Robotiq 3F + overhead camera + scene
│
├── logs/                           # Generated episode CSVs and metadata (gitignored)
├── docs/
│   └── dashboard_roadmap.md        # Full project roadmap with completion status
│
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

```powershell
# Install Python dependencies
pip install numpy

# Install Webots R2025a
# https://cyberbotics.com/
```

### Run Benchmarks (no Webots needed)

```powershell
cd C:\Users\jdmap\Desktop\6DOF-VLA-T

# Sanity check imports
python -c "from sensors.joint_state_monitor import AxisMonitor; print('imports OK')"

# Run stable grasp benchmark
python benchmarks\benchmark_stable_grasp.py
```

### Run in Webots

1. Open Webots → `File → Open World` → `worlds\ur10e_touchnet.wbt`
2. Ensure the `UR10e` node has `controller "ur10e_touchnet"` set
3. Hit play — the arm moves to ready pose and begins the pick and place demo

### Run Controller Externally (PowerShell)

```powershell
# Set Webots environment
$env:WEBOTS_HOME = "C:\Users\jdmap\AppData\Local\Programs\Webots"
$env:PYTHONPATH = "C:\Users\jdmap\AppData\Local\Programs\Webots\lib\controller\python"

# In Webots, set controller field to <extern> then hit play
# Then run from repo root:
python controllers\ur10e_touchnet\ur10e_touchnet.py
```

---

## Key Components

### Sensors

| File | Description |
|------|-------------|
| `joint_state_monitor.py` | Tracks all 6 UR10e joints each timestep. Outputs position, velocity, peak velocity, mean speed, rolling std, cumulative motion, range utilization, and active/idle/near-limit status. |
| `tactile_array_sensor.py` | Simulates a GelSight-style 16×16 taxel pressure array using the Hertz contact model. Outputs pressure map, total force, contact area, and centroid position. |
| `ft_sensor_interface.py` | Simulates a 6-axis F/T sensor with configurable Gaussian noise, slow bias drift, and a `tare()` method to zero before grasping. |

### Physics

| File | Description |
|------|-------------|
| `contact_model.py` | Hertz sphere-on-flat contact model. Computes indentation depth, contact radius, peak pressure, and slip condition via KKT complementarity. Returns `ContactState` with grip margin. |
| `material_properties.py` | Registry of 21 materials (steel, silicone, rubber, glass, egg, ice, teflon, etc.) with Young's modulus, Poisson's ratio, static and kinetic friction coefficients. |

### Objects

| File | Description |
|------|-------------|
| `object_library.py` | 22 graspable objects across 5 categories: rigid, soft, elastic, fragile, slippery. Each object has geometry, mass, size, material, and `max_safe_force` threshold. Supports `with_friction()` and `with_stiffness()` for ablation studies. |

### Pipeline

| File | Description |
|------|-------------|
| `episode_runner.py` | Orchestrates a single grasp episode. Integrates all sensors, contact model, and logger each timestep. Contains `VLAInterface` abstract class and two baseline policies. |
| `log_export.py` | Writes per-timestep rows to CSV in CPIM/TouchNet input format. Columns include joint state (6 joints), contact state, tactile summary, and F/T readings. Also writes JSON metadata per episode. |

---

## Vision-Guided Pick and Place Demo

`ur10e_touchnet.py` runs a full closed-loop demo combining vision, distance sensing, and the physics contact model:

1. **SCAN** — arm idles at ready pose, waits for the overhead camera to detect a colored object
2. **AIM** — arm rotates toward the object using world coordinates computed from the overhead camera (color blob detection → pixel-to-world projection → `atan2` pan angle)
3. **DESCEND** — arm lowers gradually while reading a wrist-mounted `RangeFinder`; stops once the object is within grasp range
4. **GRASP** — gripper closes incrementally; success is confirmed using **both** real finger position sensor feedback and the Hertz contact model's `grip_margin`
5. **LIFT → CARRY → PLACE_DESCEND → RELEASE → RETURN** — object is moved to a second table and the arm loops back to scan for the next object

Throughout `AIM` and `DESCEND`, the wrist is kept level (pointing straight down) by solving `wrist_1 = target_pitch - shoulder_lift - elbow` using **live sensor readings** each step — this decouples position control from orientation control so the gripper doesn't drift as the arm reaches and lowers.

The overhead camera runs as a second, independent Webots controller (`overhead_camera_detection.py`) and shares detected object world-coordinates with the main controller via a small JSON file in `logs/`.

---

## VLA Integration

The VLA model slot is defined in `pipeline/episode_runner.py`:

```python
class VLAInterface(ABC):
    @abstractmethod
    def predict_grasp_force(
        self,
        tactile_map: np.ndarray,   # (16, 16) float32 — taxel pressures in Pa
        joint_state: list[float],  # 6 joint positions in rad
        ft_wrench: tuple,          # (fx, fy, fz, tx, ty, tz)
        object_desc: str,          # natural language object description
    ) -> float:                    # target gripper closing force in N
        ...
```

To plug in a VLA model:

```python
class MyVLAModel(VLAInterface):
    def predict_grasp_force(self, tactile_map, joint_state, ft_wrench, object_desc):
        # your model inference here
        return predicted_force_n

config = EpisodeConfig(
    scenario="stable_grasp",
    object_name="silicone_ball",
    policy=MyVLAModel()
)
```

---

## Benchmark Results

| Benchmark | Status | Pass Rate | Notes |
|-----------|--------|-----------|-------|
| Stable Grasp | Complete | 29/30 (97%) | foam_cube at 12N correctly fails (exceeds max_safe_force) |
| Slip Detection | TODO | — | |
| Force Overshoot | TODO | — | Key benchmark for VLA pressure control |
| Generalization | TODO | — | |

---

## Object Categories

| Category | Examples | Key Property |
|----------|----------|--------------|
| Rigid | steel_bolt, aluminum_can, ceramic_mug | High E, moderate friction |
| Soft | foam_cube, silicone_ball, gel_pad | Low E, high compliance |
| Elastic | latex_glove_roll, neoprene_grip | High deformation recovery |
| Fragile | egg (4.5N limit), glass_vial (8N), silicon_wafer (3N) | Low max_safe_force |
| Slippery | teflon_rod, ice_cube, wet_bottle | mu_s < 0.12 |

---

## Notes

- `controller` module is Webots-internal — not pip installable. Add
  `C:\Users\jdmap\AppData\Local\Programs\Webots\lib\controller\python`
  to your IDE interpreter paths to suppress import warnings.
- All benchmark scripts run fully offline with simulated joint data.
- Episode logs are written to `logs/` and gitignored by default.
- The vision-guided demo requires **both** controllers running simultaneously: `ur10e_touchnet` (arm) and `overhead_camera_detection` (fixed overhead camera, defined as a separate `Robot` node in the world file).
- `logs/overhead_detections.json` is generated automatically and used to pass detected object coordinates between the two controllers — gitignored.
- See `docs/dashboard_roadmap.md` for full project status and planned features.

---

## References

Material properties, contact physics, and control concepts in this project draw on the following sources. All implementation code is original to this repository; references below informed approach and were used to validate against standard formulations.

| Topic | Source |
|-------|--------|
| Hertz contact model | K. L. Johnson, *Contact Mechanics*, Cambridge University Press, 1985 — used in `physics/contact_model.py` for indentation depth, contact radius, and peak pressure calculations |
| KKT complementarity / slip condition | D. E. Stewart & J. C. Trinkle, "An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction," *International Journal for Numerical Methods in Engineering*, 1996 — used in `physics/contact_model.py` for the Coulomb friction cone check and kinetic slip clamping |
| Complementarity contact framework (reference implementation) | [PyRoboCOP](https://github.com/merlresearch/PyRoboCOP) — Raghunathan, Jha, Romeres, MERL Research, 2022. Open-source Python package for robotic control with complementarity-constrained contact models; used as a conceptual reference for how complementarity constraints are structured in code, not directly imported |
| Inverse kinematics — decoupled position/orientation solving | [Inverse-Kinematics-6-DOF-Python-With-Visualizer](https://github.com/zahid58/Inverse-Kinematics-6-DOF-Python-With-Visualizer) — Md. Zahidul Islam (based on [mithi/arm-ik](https://github.com/mithi/arm-ik)). Demonstrates the technique of solving wrist position and end-effector orientation as separate steps for a 6DOF arm. The concept (not the code) informed `compute_level_wrist1()` in `ur10e_touchnet.py`, which solves the wrist joint angle needed to keep the gripper level using live sensor feedback |
| UR10e arm model and IK | Webots R2025a — Universal Robots UR10e proto and built-in IK solver, [cyberbotics.com](https://cyberbotics.com) |
| Material engineering data | Standard engineering references (Young's modulus, Poisson's ratio, friction coefficients) for the 21 materials in `physics/material_properties.py` |

---

## Repository
[github.com/bmap02/6DOF-VLA-T](https://github.com/bmap02/6DOF-VLA-T)