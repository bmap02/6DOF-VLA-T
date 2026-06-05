# 6DOF-VLA-T Project Roadmap
UR10e + Robotiq 3F | Webots R2025a | TouchNet / CPIM Simulation Testbed

---

## STATUS KEY
[x] Complete
[-] In progress
[ ] Not started

---

## 1. ENVIRONMENT SETUP 

### 1.1 Robot & Gripper Model
[x] UR10e arm in Webots world (ur10e_touchnet.wbt)
[x] Robotiq 3F gripper attached to toolSlot
[x] Kinematics working via Webots built-in IK
[x] Gripper open/close commands in controller
[ ] Calibrate gripper finger joint names against actual Robotiq 3F proto
[ ] Verify gripper force feedback from motor torque sensors

### 1.2 Object Library
[x] object_library.py — 22 objects across 5 categories
[x] Configurable stiffness (E), friction (mu_s, mu_k), compliance
[x] Categories: rigid, soft, elastic, fragile, slippery
[x] with_friction() and with_stiffness() ablation helpers
[ ] Expand to 50+ material categories per original spec
[ ] Add objects to Webots world scene (place on small table)
[ ] Wire Webots object physics to material_properties.py values

### 1.3 Sensor Models
[x] joint_state_monitor.py — per-joint pos/vel/peak/avg/cumul/range stats
[x] tactile_array_sensor.py — 16x16 Hertz pressure distribution (GelSight style)
[x] ft_sensor_interface.py — 6-axis F/T with noise, drift, tare()
[ ] Wire tactile sensor to actual Webots TouchSensor node
[ ] Wire F/T sensor to actual Webots force feedback from gripper
[ ] Validate simulated tactile readings against expected Hertz distribution

### 1.4 Physics Engine (PINN-Calibrated)
[x] contact_model.py — Hertz contact + KKT complementarity slip condition
[x] material_properties.py — 21 materials with E, nu, mu_s, mu_k
[x] ContactState dataclass with deformation, contact radius, pressure peak
[ ] PINN calibration pass — tune Hertz model constants against real sensor data
[ ] Add soft body / elastic deformation model for foam/gel objects
[ ] Validate contact model against known Hertz analytical solutions

---

## 2. SIMULATION PIPELINE & DATA GENERATION

### 2.1 Episode Runner
[x] episode_runner.py — full episode lifecycle (start/step/stop)
[x] State machine for pick and place demo (9 stages)
[x] VLAInterface abstract base class with predict_grasp_force()
[x] ConstantForcePolicy and ProportionalForcePolicy baselines
[ ] Wire actual Webots joint positions into runner (currently simulated FD)
[ ] Add episode reset — return arm to ready pose between episodes
[ ] Add object detection — confirm object is grasped before lift stage

### 2.2 Trajectory Logger
[x] log_export.py — CSV + JSON metadata per episode
[x] CPIM/TouchNet format columns (joint state, contact, tactile, F/T)
[x] DataLogger with start()/log()/stop() pattern
[ ] Add full 16x16 taxel map logging (currently stripped for size)
[ ] Add Webots simulation timestamp sync to logger
[ ] Export dataset in TouchNet input format spec

### 2.3 Webots Controller
[x] ur10e_touchnet.py — Webots robot controller entry point
[x] Arm moves to ready pose on startup
[x] Pick and place demo state machine running
[x] Joint monitor printing every 30 steps
[ ] Set controller to extern mode for PowerShell execution
[ ] Confirm all 6 UR10e joint motor names resolve correctly
[ ] Confirm Robotiq 3F finger motor names resolve correctly
[ ] Add real gripper force reading from motor torque feedback

---

## 3. TESTING INTERFACE

### 3.1 Benchmarks
[x] benchmark_stable_grasp.py — 29/30 passing (97%)
[ ] benchmark_slip_detection.py — slip events across friction levels
[ ] benchmark_force_overshoot.py — fragile objects (egg, glass vial, wafer)
[ ] benchmark_generalization.py — unseen objects/materials

### 3.2 Algorithm Interface (TouchNet / CPIM)
[ ] Define exact TouchNet input format schema
[ ] Write adapter that converts log_export.py CSV to TouchNet format
[ ] Run repeatable grasping experiments end-to-end
[ ] Save sensor data, actions, and outcomes per spec

---

## 4. VLA MODEL INTERFACE

### 4.1 VLA Hook (Future)
[x] VLAInterface abstract base class defined in episode_runner.py
[x] predict_grasp_force(tactile_map, joint_state, ft_wrench, object_desc) -> float
[ ] Implement VLA model subclass
[ ] Wire VLA output to gripper force command in Webots
[ ] Test VLA on stable_grasp benchmark
[ ] Test VLA on force_overshoot benchmark (fragile objects)
[ ] Evaluate: success rate, slip rate, force error metrics

---

## 5. LIVE DASHBOARD (Deferred)

### 5.1 Current Approach
[ ] Set UR10e controller to <extern> in Webots
[ ] Run controller from PowerShell with WEBOTS env vars set:
    $env:WEBOTS_HOME = "C:\Users\jdmap\AppData\Local\Programs\Webots"
    $env:PYTHONPATH = "C:\Users\jdmap\AppData\Local\Programs\Webots\lib\controller\python"
    python controllers\ur10e_touchnet\ur10e_touchnet.py
[ ] ANSI in-place table renders correctly in PowerShell terminal

### 5.2 Second Monitor Browser Dashboard (Planned)
[ ] dashboard/dashboard_server.py — asyncio HTTP + WebSocket server
[ ] dashboard/dashboard.html — live joint table, velocity bars, stage indicator
[ ] Controller spawns server subprocess on startup
[ ] Joint data streamed over WebSocket localhost:8765
[ ] HTML served at localhost:8080, open in Chrome on second monitor
[ ] Reuse ur10e_touchnet.html as base (already designed)
[ ] Dependencies: pip install websockets

---

## 6. FINAL OUTPUTS (Per Original Spec)

[ ] Documented simulation environment (this roadmap + inline code docs)
[ ] Example test scripts (benchmark_*.py files)
[ ] Dataset of simulated contact trajectories (logs/ CSV files)
[ ] Evaluation report with metrics:
    - Success rate per object category
    - Slip rate per friction level
    - Force error vs max_safe_force threshold
    - Generalization score on unseen materials

---

## 7. BENCHMARK SCENARIOS (Per Original Spec)

[ ] Stable grasping — rigid + soft objects at varied force levels (IN PROGRESS)
[ ] Slip detection and prevention — slippery category objects
[ ] Force overshoot on fragile objects — egg, glass_vial, silicon_wafer
[ ] Generalization to unseen objects/materials — held-out material categories

---

## NOTES

### What runs without Webots
- benchmark_stable_grasp.py (fully offline, simulated joint data)
- All future benchmark_*.py files (same pattern)
- object_library.py, material_properties.py, contact_model.py (pure Python)

### What requires Webots
- controllers/ur10e_touchnet/ur10e_touchnet.py (imports controller module)
- Any live sensor data collection

### Extern Controller Mode
See section 5.1 above. The controller module resolves when PYTHONPATH
points to Webots lib/controller/python. Run from repo root only.

### VLA Integration Point
See episode_runner.py -> VLAInterface
Subclass it, implement predict_grasp_force(), pass to EpisodeConfig(policy=YourModel())