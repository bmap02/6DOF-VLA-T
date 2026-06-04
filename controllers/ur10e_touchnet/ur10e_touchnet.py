"""controllers/ur10e_touchnet/ur10e_touchnet.py — UR10e TouchNet controller.

Webots robot controller for the UR10e + Robotiq 3F gripper.
Handles:
  - Arm startup pose (stands arm upright)
  - Joint state monitoring via joint_state_monitor
  - Gripper open/close commands
  - Episode runner integration for TouchNet pipeline

Set this as the UR10e robot controller in Webots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from controller import Robot, Motor, PositionSensor
from sensors.joint_state_monitor import AxisMonitor
from pipeline.episode_runner import EpisodeConfig, EpisodeRunner, ConstantForcePolicy
from objects.object_library import get as get_object
from utils.kinematics_utils import finite_diff_velocity

TIMESTEP    = 32      # ms — match world basicTimeStep
PRINT_EVERY = 30      # steps between console prints

# ── UR10e joint motor names ───────────────────────────────────────────────────
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Robotiq 3F gripper motor names
GRIPPER_JOINTS = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1",
]

# ── Startup pose (radians) — arm stands upright, ready to reach down ──────────
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
READY_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# Gripper states
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.85   # rad — adjust for object size


def move_to_pose(motors, targets, speed=0.5):
    """Command all arm joints to target positions."""
    for motor, target in zip(motors, targets):
        motor.setVelocity(speed)
        motor.setPosition(target)


def set_gripper(gripper_motors, position):
    """Open or close all gripper fingers."""
    for motor in gripper_motors:
        motor.setVelocity(0.5)
        motor.setPosition(position)


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
        if motor is None:
            print(f"[WARN] Gripper motor '{name}' not found.")
            continue
        gripper_motors.append(motor)

    # ── Axis monitor ──────────────────────────────────────────────────────────
    axis_monitor = AxisMonitor()

    # ── Episode runner ────────────────────────────────────────────────────────
    obj    = get_object("silicone_ball")
    config = EpisodeConfig(
        scenario    = "stable_grasp",
        object_name = obj.name,
        max_steps   = 800,
        policy      = ConstantForcePolicy(force_n=4.0),
    )
    runner = EpisodeRunner(config=config, robot=robot)
    runner.start(obj)

    # ── Move to ready pose on startup ─────────────────────────────────────────
    move_to_pose(arm_motors, READY_POSE, speed=0.3)
    set_gripper(gripper_motors, GRIPPER_OPEN)
    print("[ur10e_touchnet] Moving to ready pose...")

    prev_positions = [None] * len(arm_sensors)
    step = 0

    while robot.step(TIMESTEP) != -1 and step < config.max_steps:
        sim_time  = robot.getTime()

        # Read joint positions
        positions = [
            s.getValue() if s else 0.0
            for s in arm_sensors
        ]

        # Finite difference velocities
        velocities = [
            finite_diff_velocity(positions[i], prev_positions[i], TIMESTEP * 1e-3)
            for i in range(len(arm_sensors))
        ]
        prev_positions = positions[:]

        # Update axis monitor
        axis_monitor.update_all(positions, velocities)

        # Run episode step
        recommended_force = runner.step(
            positions      = positions,
            velocities     = velocities,
            gripper_force  = 4.0,
        )

        # Apply gripper force recommendation
        # Map force (N) to gripper position (rad): 0N=open, 50N=closed
        gripper_pos = min(GRIPPER_CLOSED, recommended_force / 50.0 * GRIPPER_CLOSED)
        set_gripper(gripper_motors, gripper_pos)

        # Print axis stats periodically
        if step % PRINT_EVERY == 0:
            axis_monitor.print_table(sim_time, step)

        step += 1

    runner.stop(outcome="completed", metrics={"steps": step})
    print("[ur10e_touchnet] Episode complete.")


if __name__ == "__main__":
    main()