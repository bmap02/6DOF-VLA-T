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
<img width="4959" height="5732" alt="diagram (1)" src="https://github.com/user-attachments/assets/93ad624e-e9a0-4888-be15-6739f5e2c9d4" />

```

---

## Quick Start

### Prerequisites

```powershell
# Install Python dependencies
pip install numpy drake

# Install Webots R2025a
# https://cyberbotics.com/
```

`drake` (pydrake) provides the `MultibodyPlant`/`InverseKinematics` engine that
`ur10e_touchnet.py` uses to parse the URDF models in `urdf/` and solve full
6-DOF inverse kinematics each control step.

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
3. Hit play — the arm moves to ready pose and begins the pick and lift demo

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

## Vision-Guided Pick and Lift Demo

`ur10e_touchnet.py` runs a full closed-loop demo combining vision, distance sensing, and Drake-based inverse kinematics:

1. **INITIALIZE** — arm drives to a fixed ready pose
2. **TRANSIT_UP** — arm rises straight up to a safe transit height above the workspace, then snapshots the overhead camera's current object detection (world x/y + a stable Recognition `id`) before the arm's own body can enter the overhead camera's view and be mistaken for an object
3. **TRANSIT_OVER → MOVE_TO_REGION** — arm moves horizontally to the locked region, then descends to an approach hover height directly above it
4. **CAMERA_TRACK_ALIGN** — the wrist camera looks up the same tracked object `id` via Webots Camera Recognition and the arm servos its XY position until it's centered directly above the object; the object's recognized world Z is captured as the grasp depth target
5. **VERTICAL_DESCEND** — arm lowers gradually, primarily targeting the object's own recognized height, with a wrist-mounted `RangeFinder` reading kept as a secondary/earlier trigger
6. **CLOSE_GRIPPER** — gripper closes onto the object
7. **LIFT** — arm rises back to transit height with the object grasped, then returns to **IDLE**

Both the overhead camera and the wrist camera use [Webots Camera Recognition](https://cyberbotics.com/doc/reference/recognition) rather than color-blob pixel scanning — this gives ground-truth object positions and a stable per-object id, so "the object to pick" is whatever is physically inside the `ObjectBoundingBox` region marker (defined in the world file) rather than a hardcoded color or position. Inverse kinematics is solved with Drake's `InverseKinematics` optimizer over the UR10e + Robotiq 3F gripper URDF models (`urdf/`), jointly constraining palm position and orientation each step rather than a decoupled per-joint solve.

The overhead camera runs as a second, independent Webots controller (`overhead_camera_detection.py`) and shares its detected object's world-coordinates and Recognition `id` with the main controller via a small JSON file in `logs/`.

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
- `logs/overhead_detections.json` is generated automatically and used to pass the tracked object's world coordinates and Recognition `id` between the two controllers — gitignored.

---

## References

Material properties, contact physics, and control concepts in this project draw on the following sources. All implementation code is original to this repository; references below informed approach and were used to validate against standard formulations.

| Topic | Source |
|-------|--------|
| Hertz contact model | K. L. Johnson, *Contact Mechanics*, Cambridge University Press, 1985 — used in `physics/contact_model.py` for indentation depth, contact radius, and peak pressure calculations |
| KKT complementarity / slip condition | D. E. Stewart & J. C. Trinkle, "An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction," *International Journal for Numerical Methods in Engineering*, 1996 — used in `physics/contact_model.py` for the Coulomb friction cone check and kinetic slip clamping |
| Complementarity contact framework (reference implementation) | [PyRoboCOP](https://github.com/merlresearch/PyRoboCOP) — Raghunathan, Jha, Romeres, MERL Research, 2022. Open-source Python package for robotic control with complementarity-constrained contact models; used as a conceptual reference for how complementarity constraints are structured in code, not directly imported |
| Multibody dynamics & numerical inverse kinematics | Russ Tedrake, [*Underactuated Robotics*](https://underactuated.csail.mit.edu/index.html), MIT — course notes underpinning the Drake `MultibodyPlant` formulation and optimization-based IK used throughout `compute_ik()` in `ur10e_touchnet.py` |
| Manipulation, grasping, and perception-driven pick pipelines | Russ Tedrake, [*Robotic Manipulation*](https://manipulation.mit.edu/robot.html), MIT — informed the region-locked, recognition-driven pick sequence (`TRANSIT_UP` → `CAMERA_TRACK_ALIGN` → `VERTICAL_DESCEND`) and the use of Drake's `InverseKinematics` solver over a URDF-parsed arm + gripper model |
| Object detection (world coordinates + stable identity) | [Webots Camera Recognition](https://cyberbotics.com/doc/reference/recognition), R2025a — ground-truth object position and per-object `id`, used in both `overhead_camera_detection.py` and `ur10e_touchnet.py` in place of color-blob pixel scanning |
| UR10e arm model | Webots R2025a — Universal Robots UR10e proto, [cyberbotics.com](https://cyberbotics.com) |
| Material engineering data | Standard engineering references (Young's modulus, Poisson's ratio, friction coefficients) for the 21 materials in `physics/material_properties.py` |

---

## Repository
[github.com/bmap02/6DOF-VLA-T](https://github.com/bmap02/6DOF-VLA-T)
