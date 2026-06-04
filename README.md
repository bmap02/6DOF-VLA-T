# TouchNet Simulation Testbed
ABB IRB 4600 · Webots R2023b+ · Python 3.10+

## Structure
```
touchnet_testbed/
├── controllers/
│   └── abb_touchnet/
│       └── abb_touchnet.py     # Webots robot controller (entry point)
├── sensors/
│   ├── axis_monitor.py         # Per-joint stats + CSV logging
│   ├── tactile_sensor.py       # Simulated GelSight / DIGIT tactile array
│   └── force_torque_sensor.py  # 6-axis F/T sensor
├── physics/
│   ├── contact_model.py        # Hertz contact + KKT constraints
│   └── material_properties.py  # Shared material constants
├── objects/
│   └── object_library.py       # 50+ material categories, configurable props
├── pipeline/
│   ├── episode_runner.py       # Run + log a single grasp episode
│   └── data_logger.py          # CPIM/TouchNet format trajectory writer
├── benchmarks/
│   ├── stable_grasp.py         # Benchmark: stable grasping
│   ├── slip_detection.py       # Benchmark: slip detection + prevention
│   ├── force_overshoot.py      # Benchmark: fragile object force overshoot
│   └── generalization.py       # Benchmark: unseen object/material
├── utils/
│   └── math_utils.py           # Shared vector/rotation helpers
├── logs/                        # CSV + JSON trajectory output (gitignored)
└── README.md
```

## Quick start
1. Copy `controllers/abb_touchnet/` into your Webots project controllers folder.
2. Set the ABB robot's controller to `abb_touchnet` in the scene tree.
3. Run a benchmark:  `python benchmarks/stable_grasp.py`

## VLA model hook
See `pipeline/episode_runner.py` → `VLAInterface` stub.
Implement `predict_grasp_force(tactile, joint_state) -> float` and wire it in.