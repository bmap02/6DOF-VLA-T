# ur10e_touchnet.py - UR10e TouchNet controller (clean rebuild)
#
# Simplified architecture:
#   1. Overhead camera gives ROUGH object position (pan angle only)
#   2. Arm pans toward the object using overhead feedback
#   3. Arm descends slowly while reading the wrist-mounted RangeFinder
#   4. Descend stops once range finder reports the object is close
#   5. Gripper closes, contact model checks grip security
#   6. Lift, carry to wood table, place, return to ready
#
# This replaces the earlier wrist-camera pixel-tracking approach, which
# had too many fragile interactions. Distance sensing is more robust
# for "how close am I to the table/object" than vision-based centering.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from controller import Robot, Motor, PositionSensor, Camera, RangeFinder
from sensors.joint_state_monitor import AxisMonitor
from physics.contact_model import HertzContact
from physics.material_properties import get as get_material
from utils.kinematics_utils import finite_diff_velocity, clamp
import json
import math

TIMESTEP = 32

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_JOINTS = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1",
]

GRIPPER_OPEN   = 0.05    # Robotiq 3F valid minimum is ~0.0495
GRIPPER_CLOSED = 0.85

# ── Overhead detection (rough position only) ─────────────────────────────────
TARGET_COLORS = {
    "red":    {"rgb": (255, 0,   0),   "tol": 70, "material": "silicone"},
    "blue":   {"rgb": (85,  170, 255), "tol": 70, "material": "rubber"},
    "yellow": {"rgb": (255, 255, 0),   "tol": 70, "material": "wood_pine"},
}

OVERHEAD_DETECTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "overhead_detections.json"
)

ARM_BASE_X = -4.84
ARM_BASE_Y = -0.35
ARM_BASE_ROTATION = 1.5708


def read_overhead_detections():
    try:
        with open(OVERHEAD_DETECTIONS_PATH, "r") as f:
            data = json.load(f)
        return data.get("detections", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def world_to_pan_angle(target_x, target_y):
    dx = target_x - ARM_BASE_X
    dy = target_y - ARM_BASE_Y
    raw_angle = math.atan2(dy, dx)
    pan = raw_angle - ARM_BASE_ROTATION
    while pan > math.pi:
        pan -= 2 * math.pi
    while pan < -math.pi:
        pan += 2 * math.pi
    return pan


# ── Poses (rad) [pan, lift, elbow, w1, w2, w3] ───────────────────────────────
POSES = {
    "ready": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    "scan":  [0.0, -1.7000, 1.9000, -1.7708, -1.5708, 0.0],   # pan overridden per-target
    "place": [1.0, -1.4000, 1.5000, -1.6000, -1.5708, 0.0],
}

# How far the table is below the range finder reading threshold, in meters.
# Tune this after watching the first few runs.
GRASP_DISTANCE_THRESHOLD = 0.06   # stop descending once object is within 6cm
DESCEND_STEP_LIFT  = 0.01         # how much to lower shoulder_lift each step
DESCEND_STEP_ELBOW = 0.01         # how much to extend elbow each step (compensate)
MAX_DESCEND_STEPS  = 150


def move_to_pose(motors, targets, speed=0.4):
    for motor, target in zip(motors, targets):
        motor.setVelocity(speed)
        motor.setPosition(target)


def set_gripper(gripper_motors, position, speed=0.3):
    for motor in gripper_motors:
        motor.setVelocity(speed)
        motor.setPosition(position)


# Target gripper pitch in radians, measured as the sum of
# shoulder_lift + elbow + wrist_1. -pi/2 (-1.5708) corresponds to the
# gripper pointing straight down in this arm's convention -- matches the
# wrist_1 value used in the original "ready"/"scan" poses.
TARGET_GRIPPER_PITCH = -1.5708


def compute_level_wrist1(live_shoulder_lift, live_elbow):
    """
    Solve for the wrist_1 angle that keeps the gripper pointing straight
    down, using LIVE sensor feedback for shoulder_lift and elbow rather
    than commanded targets. This actively corrects drift instead of
    applying a fixed offset.

    Gripper pitch in the vertical plane is the running sum of joint
    angles along the arm: shoulder_lift + elbow + wrist_1. Solving for
    wrist_1 given the other two (read live from sensors) keeps the
    gripper level regardless of how far the arm is reaching/descending:
        wrist_1 = target_pitch - shoulder_lift - elbow
    """
    return TARGET_GRIPPER_PITCH - live_shoulder_lift - live_elbow


def joints_reached(sensors, targets, tol=0.05):
    for s, t in zip(sensors, targets):
        if s and abs(s.getValue() - t) > tol:
            return False
    return True


def read_center_distance(range_finder):
    """Read the center pixel of the range finder's depth image."""
    width  = range_finder.getWidth()
    height = range_finder.getHeight()
    depth  = range_finder.getRangeImage()
    if depth is None:
        return None
    idx = (height // 2) * width + (width // 2)
    val = depth[idx]
    if math.isinf(val) or math.isnan(val):
        return None
    return val


def detect_blobs(camera):
    MIN_PIXELS = 15
    SAMPLE_STEP = 4
    width  = camera.getWidth()
    height = camera.getHeight()
    image  = camera.getImage()
    if image is None:
        return {}
    sums = {name: [0, 0, 0] for name in TARGET_COLORS}
    for y in range(0, height, SAMPLE_STEP):
        for x in range(0, width, SAMPLE_STEP):
            r = Camera.imageGetRed(image, width, x, y)
            g = Camera.imageGetGreen(image, width, x, y)
            b = Camera.imageGetBlue(image, width, x, y)
            for name, spec in TARGET_COLORS.items():
                tr, tg, tb = spec["rgb"]
                tol = spec["tol"]
                if abs(r - tr) < tol and abs(g - tg) < tol and abs(b - tb) < tol:
                    sums[name][0] += x
                    sums[name][1] += y
                    sums[name][2] += 1
    results = {}
    for name, (sx, sy, count) in sums.items():
        if count >= MIN_PIXELS // SAMPLE_STEP:
            results[name] = (sx / count, sy / count, count)
    return results


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
            sensor = motor.getPositionSensor()
            if sensor:
                sensor.enable(TIMESTEP)
            gripper_motors.append(motor)
        else:
            print(f"[WARN] Gripper motor '{name}' not found.")

    # ── Wrist camera (rough alignment cue only, not used for fine centering) ──
    camera = robot.getDevice("wrist_camera")
    if camera:
        camera.enable(TIMESTEP)

    # ── Range finder (distance sensing for descend) ────────────────────────────
    range_finder = robot.getDevice("range-finder")
    if range_finder is None:
        print("[ERROR] 'range-finder' device not found. Check world file device name.")
        return
    range_finder.enable(TIMESTEP)
    print("[grasp_demo] Range finder enabled.")

    axis_monitor  = AxisMonitor()
    contact_model = HertzContact(finger_radius=0.010)

    # ── State machine: SCAN -> AIM -> DESCEND -> GRASP -> LIFT -> CARRY ->
    #                   PLACE_DESCEND -> RELEASE -> RETURN -> SCAN ──────────────
    state = "SCAN"
    hold_counter = 0
    grasp_target = None
    descend_steps = 0
    grasp_close_step = 0

    move_to_pose(arm_motors, POSES["ready"])
    print("[grasp_demo] State: SCAN -- arm idling upright, waiting for overhead detection")

    prev_positions = [None] * len(arm_sensors)
    step = 0
    max_steps = 6000

    while robot.step(TIMESTEP) != -1 and step < max_steps:
        sim_time = robot.getTime()
        positions = [s.getValue() if s else 0.0 for s in arm_sensors]
        velocities = [
            finite_diff_velocity(positions[i], prev_positions[i], TIMESTEP * 1e-3)
            for i in range(len(arm_sensors))
        ]
        prev_positions = positions[:]
        axis_monitor.update_all(positions, velocities)

        # ── SCAN ──────────────────────────────────────────────────────────────
        if state == "SCAN":
            move_to_pose(arm_motors, POSES["ready"], speed=0.3)
            overhead = read_overhead_detections()

            if overhead and joints_reached(arm_sensors, POSES["ready"], tol=0.1):
                best_name, best_dist = None, float("inf")
                for name, det in overhead.items():
                    d = ((det["world_x"] - ARM_BASE_X) ** 2 +
                         (det["world_y"] - ARM_BASE_Y) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best_name = d, name
                grasp_target = best_name
                print(f"[grasp_demo] Target acquired: '{grasp_target}'. Aiming.")
                state = "AIM"
                hold_counter = 0
            else:
                hold_counter += 1
                if hold_counter > 200:
                    print("[grasp_demo] No objects detected. Finishing.")
                    state = "FINISHED"
                    hold_counter = 0

        # ── AIM: rotate base toward rough overhead position ─────────────────────
        elif state == "AIM":
            overhead = read_overhead_detections()
            if grasp_target not in overhead:
                hold_counter += 1
                if hold_counter > 60:
                    print(f"[grasp_demo] Lost '{grasp_target}' during aim. Re-scanning.")
                    state = "SCAN"
                    hold_counter = 0
            else:
                target_world = overhead[grasp_target]
                pan_angle = world_to_pan_angle(target_world["world_x"], target_world["world_y"])
                planar_dist = ((target_world["world_x"] - ARM_BASE_X) ** 2 +
                                (target_world["world_y"] - ARM_BASE_Y) ** 2) ** 0.5

                # Rough reach adjustment so the gripper lands ON the object,
                # not past it. Scaled gently around the default scan pose --
                # this is intentionally approximate, the RangeFinder in
                # DESCEND handles final precision.
                reach_adjust = clamp((planar_dist - 0.866) * 0.15, -0.10, 0.10)

                aim_pose = POSES["scan"][:]
                aim_pose[0] = pan_angle
                aim_pose[1] -= reach_adjust
                aim_pose[2] += reach_adjust

                # Keep gripper level using LIVE sensor feedback for the
                # actual current shoulder_lift/elbow values, not the
                # commanded target -- this actively corrects based on
                # where the arm really is each step.
                aim_pose[3] = compute_level_wrist1(positions[1], positions[2])
                aim_pose[4] = POSES["scan"][4]
                aim_pose[5] = POSES["scan"][5]
                move_to_pose(arm_motors, aim_pose, speed=0.3)

                if step % 30 == 0:
                    print(f"[grasp_demo] AIM: pan_angle={pan_angle:.3f} cur_pan={positions[0]:.3f} "
                          f"wrist1_target={aim_pose[3]:.3f}")

                if joints_reached(arm_sensors, aim_pose, tol=0.05):
                    hold_counter += 1
                    if hold_counter >= 20:
                        print(f"[grasp_demo] Aim complete. Beginning distance-guided descend.")
                        state = "DESCEND"
                        hold_counter = 0
                        descend_steps = 0
                else:
                    hold_counter = 0

        # ── DESCEND: lower gradually using RangeFinder feedback ─────────────────
        elif state == "DESCEND":
            set_gripper(gripper_motors, GRIPPER_OPEN)
            distance = read_center_distance(range_finder)

            if step % 15 == 0:
                print(f"[grasp_demo] DESCEND: distance={distance} step={descend_steps}")

            if distance is not None and distance <= GRASP_DISTANCE_THRESHOLD:
                print(f"[grasp_demo] Object within grasp range (distance={distance:.3f}m). Closing gripper.")
                state = "GRASP"
                hold_counter = 0
                grasp_close_step = 0
            elif descend_steps >= MAX_DESCEND_STEPS:
                print(f"[grasp_demo] Max descend reached without close range reading "
                      f"(last distance={distance}). Attempting grasp anyway.")
                state = "GRASP"
                hold_counter = 0
                grasp_close_step = 0
            else:
                cur_pose = positions[:]
                cur_pose[1] -= DESCEND_STEP_LIFT
                cur_pose[2] += DESCEND_STEP_ELBOW

                # Use LIVE sensor feedback (not commanded values) to solve
                # for the wrist_1 angle that keeps the gripper level, so
                # any accumulated drift during descend is actively corrected
                # each step rather than compounding.
                cur_pose[3] = compute_level_wrist1(positions[1], positions[2])
                cur_pose[4] = POSES["scan"][4]
                cur_pose[5] = POSES["scan"][5]
                move_to_pose(arm_motors, cur_pose, speed=0.15)
                descend_steps += 1

                if step % 15 == 0:
                    print(f"[grasp_demo] DESCEND wrist1_target={cur_pose[3]:.3f} "
                          f"(live lift={positions[1]:.3f} elbow={positions[2]:.3f})")

        # ── GRASP: close gripper, verify via position sensors + contact model ──
        elif state == "GRASP":
            grasp_close_step += 1
            GRIPPER_MIN_VALID = 0.0495
            close_pos = min(GRIPPER_CLOSED, GRIPPER_MIN_VALID + grasp_close_step * 0.01)
            set_gripper(gripper_motors, close_pos, speed=0.15)

            finger_positions = [
                m.getPositionSensor().getValue() if m.getPositionSensor() else None
                for m in gripper_motors
            ]
            fingers_responding = all(
                p is not None and p >= GRIPPER_MIN_VALID * 0.8
                for p in finger_positions
            ) if grasp_close_step > 5 else False

            gripper_force = min(8.0, grasp_close_step * 0.15)
            spec = TARGET_COLORS[grasp_target]
            material = get_material(spec["material"])
            contact = contact_model.compute(
                normal_force=gripper_force,
                tangential_force=gripper_force * 0.1,
                slip_velocity=0.0,
                mat_obj=material,
            )

            if step % 10 == 0:
                print(f"[grasp_demo] GRASP step={grasp_close_step} close_pos={close_pos:.4f} "
                      f"finger_pos={[round(p,4) if p is not None else None for p in finger_positions]} "
                      f"responding={fingers_responding}")

            if fingers_responding and contact.in_contact and contact.grip_margin > 0.3 and grasp_close_step > 15:
                print(f"[grasp_demo] Secure grip CONFIRMED on '{grasp_target}'.")
                state = "LIFT"
                hold_counter = 0
            elif grasp_close_step > 80:
                print(f"[grasp_demo] Max grasp attempts reached. Lifting anyway.")
                state = "LIFT"
                hold_counter = 0

        # ── LIFT ──────────────────────────────────────────────────────────────
        elif state == "LIFT":
            lift_pose = positions[:]
            lift_pose[1] += 0.3
            move_to_pose(arm_motors, lift_pose, speed=0.25)
            hold_counter += 1
            if hold_counter > 60:
                print(f"[grasp_demo] Lift complete. Carrying '{grasp_target}' to wood table.")
                state = "CARRY"
                hold_counter = 0

        # ── CARRY ─────────────────────────────────────────────────────────────
        elif state == "CARRY":
            move_to_pose(arm_motors, POSES["place"], speed=0.25)
            if joints_reached(arm_sensors, POSES["place"], tol=0.08):
                hold_counter += 1
                if hold_counter > 40:
                    print("[grasp_demo] Over wood table. Descending to place.")
                    state = "PLACE_DESCEND"
                    hold_counter = 0

        # ── PLACE_DESCEND ─────────────────────────────────────────────────────
        elif state == "PLACE_DESCEND":
            place_descend = POSES["place"][:]
            place_descend[1] -= 0.18
            place_descend[2] += 0.18
            move_to_pose(arm_motors, place_descend, speed=0.15)
            hold_counter += 1
            if hold_counter > 60:
                state = "RELEASE"
                hold_counter = 0
                print("[grasp_demo] State: RELEASE -- opening gripper")

        # ── RELEASE ───────────────────────────────────────────────────────────
        elif state == "RELEASE":
            set_gripper(gripper_motors, GRIPPER_OPEN)
            hold_counter += 1
            if hold_counter > 40:
                print(f"[grasp_demo] '{grasp_target}' placed.")
                state = "RETURN"
                hold_counter = 0

        # ── RETURN ────────────────────────────────────────────────────────────
        elif state == "RETURN":
            move_to_pose(arm_motors, POSES["ready"])
            hold_counter += 1
            if hold_counter > 50:
                print("[grasp_demo] Returning to SCAN for next object.")
                state = "SCAN"
                hold_counter = 0
                grasp_target = None

        # ── FINISHED ──────────────────────────────────────────────────────────
        elif state == "FINISHED":
            move_to_pose(arm_motors, POSES["ready"])
            set_gripper(gripper_motors, GRIPPER_OPEN)
            if step % 90 == 0:
                print("[grasp_demo] No more objects. Demo finished.")

        if step % 30 == 0:
            axis_monitor.print_table(sim_time, step)

        step += 1

    print("[grasp_demo] Episode loop ended.")


if __name__ == "__main__":
    main()