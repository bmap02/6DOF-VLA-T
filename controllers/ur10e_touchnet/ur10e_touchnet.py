# ur10e_touchnet.py - UR10e TouchNet controller
# NOTE: 'controller' is a Webots-internal module, not pip installable.
# It resolves automatically when Webots runs this file.
# To suppress IDE warnings add Webots controller path to your interpreter.
#
# Full pick and place demo sequence:
#   1. Move to ready pose (arm upright)
#   2. Open gripper
#   3. Move to approach pose above small table
#   4. Descend to grasp pose
#   5. Close gripper on object
#   6. Lift object
#   7. Move to place pose above big table
#   8. Descend to place
#   9. Open gripper
#  10. Return to ready pose
#  11. Repeat

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from controller import Robot, Motor, PositionSensor
from sensors.joint_state_monitor import AxisMonitor
from pipeline.episode_runner import EpisodeConfig, EpisodeRunner, ConstantForcePolicy
from objects.object_library import get as get_object
from utils.kinematics_utils import finite_diff_velocity

TIMESTEP    = 32

# ── UR10e joint motor names ───────────────────────────────────────────────────
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Robotiq 3F gripper motors
GRIPPER_JOINTS = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1",
]

GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.85

# ── Demo poses (rad) [pan, lift, elbow, w1, w2, w3] ──────────────────────────
POSES = {
    "ready":    [ 0.0,    -1.5708,  1.5708, -1.5708, -1.5708,  0.0   ],
    "approach": [ 1.5708, -1.2000,  1.2000, -1.5708, -1.5708,  0.0   ],
    "grasp":    [ 1.5708, -1.5000,  1.7000, -1.7000, -1.5708,  0.0   ],
    "lift":     [ 1.5708, -1.2000,  1.2000, -1.5708, -1.5708,  0.0   ],
    "place":    [ 0.0,    -1.2000,  1.2000, -1.5708, -1.5708,  0.0   ],
    "place_dn": [ 0.0,    -1.5000,  1.7000, -1.7000, -1.5708,  0.0   ],
}

# Steps to hold each pose before transitioning
HOLD_STEPS = {
    "ready":    60,
    "approach": 50,
    "grasp":    80,
    "lift":     50,
    "place":    50,
    "place_dn": 80,
}


def move_to_pose(motors, targets, speed=0.5):
    for motor, target in zip(motors, targets):
        motor.setVelocity(speed)
        motor.setPosition(target)


def set_gripper(gripper_motors, position):
    for motor in gripper_motors:
        motor.setVelocity(0.5)
        motor.setPosition(position)


def joints_reached(sensors, targets, tol=0.05):
    """Check if all joints are within tolerance of targets."""
    for s, t in zip(sensors, targets):
        if s and abs(s.getValue() - t) > tol:
            return False
    return True


def main():
    robot = Robot()

    # ── Arm motors + sensors ──────────────────────────────────────────────────
    arm_motors  = []
    arm_sensors = []

    for name in ARM_JOINTS:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"[WARN] Arm motor '{name}' not found.")
            continue
        sensor = robot.getDevice(name + "_sensor")
        if sensor is None:
            sensor = motor.getPositionSensor()
        if sensor:
            sensor.enable(TIMESTEP)
        arm_motors.append(motor)
        arm_sensors.append(sensor)

    # ── Gripper motors ────────────────────────────────────────────────────────
    gripper_motors = []
    for name in GRIPPER_JOINTS:
        motor = robot.getDevice(name)
        if motor:
            gripper_motors.append(motor)
        else:
            print(f"[WARN] Gripper motor '{name}' not found.")

    # ── Axis monitor ──────────────────────────────────────────────────────────
    axis_monitor = AxisMonitor()

    # ── Episode runner ────────────────────────────────────────────────────────
    obj    = get_object("silicone_ball")
    config = EpisodeConfig(
        scenario    = "pick_and_place",
        object_name = obj.name,
        max_steps   = 2000,
        policy      = ConstantForcePolicy(force_n=4.0),
    )
    runner = EpisodeRunner(config=config, robot=robot)
    runner.start(obj)

    # ── State machine ─────────────────────────────────────────────────────────
    demo_sequence = [
        ("ready",    GRIPPER_OPEN),
        ("approach", GRIPPER_OPEN),
        ("grasp",    GRIPPER_OPEN),
        ("grasp",    GRIPPER_CLOSED),   # close on object
        ("lift",     GRIPPER_CLOSED),
        ("place",    GRIPPER_CLOSED),
        ("place_dn", GRIPPER_CLOSED),
        ("place_dn", GRIPPER_OPEN),     # release
        ("ready",    GRIPPER_OPEN),
    ]

    current_stage  = 0
    hold_counter   = 0
    prev_positions = [None] * len(arm_sensors)
    step           = 0

    # Start first pose
    pose_name, grip = demo_sequence[current_stage]
    move_to_pose(arm_motors, POSES[pose_name])
    set_gripper(gripper_motors, grip)
    print(f"[Demo] Stage 0: {pose_name} | gripper={'open' if grip == GRIPPER_OPEN else 'closed'}")

    while robot.step(TIMESTEP) != -1 and step < config.max_steps:
        sim_time = robot.getTime()

        positions  = [s.getValue() if s else 0.0 for s in arm_sensors]
        velocities = [
            finite_diff_velocity(positions[i], prev_positions[i], TIMESTEP * 1e-3)
            for i in range(len(arm_sensors))
        ]
        prev_positions = positions[:]

        axis_monitor.update_all(positions, velocities)

        # Episode runner step
        runner.step(
            positions     = positions,
            velocities    = velocities,
            gripper_force = 4.0 if demo_sequence[current_stage][1] == GRIPPER_CLOSED else 0.0,
        )

        # ── State machine transition ──────────────────────────────────────────
        pose_name, grip = demo_sequence[current_stage]
        if joints_reached(arm_sensors, POSES[pose_name]):
            hold_counter += 1
            if hold_counter >= HOLD_STEPS.get(pose_name, 50):
                hold_counter   = 0
                current_stage  = (current_stage + 1) % len(demo_sequence)
                pose_name, grip = demo_sequence[current_stage]
                move_to_pose(arm_motors, POSES[pose_name])
                set_gripper(gripper_motors, grip)
                print(f"[Demo] Stage {current_stage}: {pose_name} | "
                      f"gripper={'open' if grip == GRIPPER_OPEN else 'closed'}")

        # Print axis stats every 30 steps
        if step % 30 == 0:
            axis_monitor.print_table(sim_time, step)

        step += 1

    runner.stop(outcome="completed", metrics={"steps": step})
    print("[ur10e_touchnet] Demo complete.")


if __name__ == "__main__":
    main()