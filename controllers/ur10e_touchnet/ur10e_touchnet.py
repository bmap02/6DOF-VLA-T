import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import numpy as np
from controller import Robot, Camera, RangeFinder

# --- Drake Kinematics & Optimization Engine ---
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve
from pydrake.multibody.inverse_kinematics import InverseKinematics

# --- Custom Subsystem Monitors & Math Extensions ---
from sensors.joint_state_monitor import AxisMonitor
from physics.contact_model import HertzContact
from utils.kinematics_utils import finite_diff_velocity


class RobotController:
    """Core control node managing closed-loop robotic manipulation.

    This class coordinates camera-driven visual servoing, distance-gated
    vertical descents, and task-space inverse kinematics using an in-memory
    Drake multibody engine tied to a Webots simulation environment.
    """

    def __init__(self):
        # 1. Initialize Webots Simulation Connection
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Hardware Configuration Mappings
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.gripper_names = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]

        # 2. Bind Webots Actuators & Sensors
        self.motors = [self.robot.getDevice(name) for name in self.joint_names]
        self.sensors = [motor.getPositionSensor() for motor in self.motors]
        for s in self.sensors:
            if s:
                s.enable(self.timestep)

        self.gripper_motors = [self.robot.getDevice(name) for name in self.gripper_names]
        for gm in self.gripper_motors:
            if gm and gm.getPositionSensor():
                gm.getPositionSensor().enable(self.timestep)

        # Bind Perception Units
        self.range_finder = self.robot.getDevice("range-finder")
        if self.range_finder:
            self.range_finder.enable(self.timestep)
        self.camera = self.robot.getDevice("wrist_camera")
        if self.camera:
            self.camera.enable(self.timestep)

        # 3. Instantiate In-Memory Drake Kinematic Engine
        self.plant, self.arm_model, self.gripper_model = self._build_drake_engine()
        self.context = self.plant.CreateDefaultContext()

        # 4. Initialize Core Monitors & Models
        self.axis_monitor = AxisMonitor()
        self.contact_model = HertzContact(finger_radius=0.010)

        # Operational State Machine Variables
        self.state = "INITIALIZE"
        self.grasp_target = "red"

        self.prev_positions = [None] * len(self.sensors)
        self.current_descend_z = 0.45
        self.locked_target_xy = None
        self.calculated_yaw = 0.0

        # Execution Guard Variables
        self.state_counter = 0
        self.MIN_Z_SAFETY_LIMIT = 0.15

        # Webots Base Anchor Frame Transformation Offsets
        self.ARM_BASE_X = -4.84
        self.ARM_BASE_Y = -0.35
        self.ARM_BASE_ROT = 1.5708

        # Color Threshold Mapping Metrics
        self.TARGET_COLORS = {
            "red": {"rgb": (255, 0, 0), "tol": 70},
            "blue": {"rgb": (85, 170, 255), "tol": 70},
            "yellow": {"rgb": (255, 255, 0), "tol": 70},
        }

        # Calibrated Sensor Threshold
        self.GRASP_DISTANCE_THRESHOLD = 0.06

    def _build_drake_engine(self):
        plant = MultibodyPlant(time_step=0.0)
        parser = Parser(plant)
        urdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "urdf"))
        arm_urdf = os.path.join(urdf_dir, "ur10e.urdf")
        gripper_urdf = os.path.join(urdf_dir, "robotiq_3f_gripper.urdf")

        if not os.path.exists(arm_urdf) or not os.path.exists(gripper_urdf):
            raise FileNotFoundError("URDF dependencies missing from project path mappings.")

        arm_model = parser.AddModels(url=f"file://{arm_urdf}")[0]
        gripper_model = parser.AddModels(url=f"file://{gripper_urdf}")[0]

        frame_tool0 = plant.GetFrameByName("tool0", arm_model)
        frame_palm = plant.GetFrameByName("palm", gripper_model)
        plant.WeldFrames(frame_tool0, frame_palm, RigidTransform())
        plant.Finalize()
        return plant, arm_model, gripper_model

    def compute_ik(self, target_xyz_world, target_rpy_world, current_q):
        """Computes inverse kinematics for the end-effector using Drake.

        Args:
            target_xyz_world (np.ndarray): Absolute 3D target coordinates.
            target_rpy_world (np.ndarray): Target orientation in Roll-Pitch-Yaw.
            current_q (list): Current 6-DOF joint angles for initial seeding.

        Returns:
            list[float] or None: Optimized 6 joint positions if valid solution
                found; None if the optimization fails or constraints break.

        Contracts:
            Pre-conditions:
                - target_xyz_world must be an array or list of length 3.
                - target_rpy_world must be an array or list of length 3.
            Post-conditions:
                - Returned joint positions map within configuration bounds.
            Invariants:
                - The static physical relative offset to ARM_BASE remains constant.
        """
        ik = InverseKinematics(self.plant, self.context)
        frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
        frame_world = self.plant.world_frame()

        x_rel = target_xyz_world[0] - self.ARM_BASE_X
        y_rel = target_xyz_world[1] - self.ARM_BASE_Y
        z_rel = target_xyz_world[2]
        target_xyz_rel = np.array([x_rel, y_rel, z_rel])

        ik.AddPositionConstraint(
            frame_palm, np.zeros(3),
            frame_world, target_xyz_rel - 0.002, target_xyz_rel + 0.002
        )
        ik.AddOrientationConstraint(
            frame_world, RotationMatrix(RollPitchYaw(target_rpy_world)),
            frame_palm, RotationMatrix(), 0.01
        )

        prog = ik.get_mutable_prog()
        q_vars = ik.q()
        for i in range(min(len(current_q), len(q_vars))):
            prog.SetInitialGuess(q_vars[i], current_q[i])

        result = Solve(prog)
        if result.is_success():
            return list(result.GetSolution(q_vars)[:6])
        return None

    def process_hand_camera_blobs(self):
        """Extracts the center pixel centroid of detected color features.

        Returns:
            tuple(float, float, int) or None: Coordinates (x, y) and absolute
                pixel area count if an asset matches; None if no signature found.

        Contracts:
            Pre-conditions:
                - Camera hardware device must be active and enabled.
                - self.grasp_target must be defined in self.TARGET_COLORS.
            Post-conditions:
                - Returned area index count must be strictly greater than 5 pixels.
        """
        if not self.camera or self.grasp_target is None:
            return None
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image = self.camera.getImage()
        if image is None:
            return None

        spec = self.TARGET_COLORS.get(self.grasp_target, {"rgb": (255, 0, 0), "tol": 70})
        tr, tg, tb = spec["rgb"]
        tol = spec["tol"]

        sum_x, sum_y, count = 0, 0, 0
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                r = Camera.imageGetRed(image, width, x, y)
                g = Camera.imageGetGreen(image, width, x, y)
                b = Camera.imageGetBlue(image, width, x, y)
                if abs(r - tr) < tol and abs(g - tg) < tol and abs(b - tb) < tol:
                    sum_x += x
                    sum_y += y
                    count += 1
        if count > 5:
            return (sum_x / count, sum_y / count, count)
        return None

    def read_range_distance(self):
        """Polls raw center-pixel flight distances from the rangefinder matrix.

        Returns:
            float or None: Value in meters down to intersection layer;
                None if out of bounds or reading evaluates to NaN/Inf.

        Contracts:
            Pre-conditions:
                - Rangefinder node must be initialized and rotated downward.
            Post-conditions:
                - Returned telemetry is guaranteed to be a real number scalar.
        """
        if not self.range_finder:
            return None
        w = self.range_finder.getWidth()
        h = self.range_finder.getHeight()
        img = self.range_finder.getRangeImage()
        if img is None:
            return None
        val = img[(h // 2) * w + (w // 2)]
        return None if (math.isinf(val) or math.isnan(val)) else val

    def update_hardware_readings(self):
        """Calculates instantaneous telemetry streams from raw physical registers.

        Returns:
            tuple(list, list): Element-wise joint angular positions (rad) and
                joint angular velocity vectors (rad/s).

        Contracts:
            Post-conditions:
                - Dimensions of returned position and velocity matrices match.
            Side Effects:
                - Mutates self.prev_positions tracking cache.
                - Pushes telemetry matrix to external self.axis_monitor node.
        """
        positions = [s.getValue() if s else 0.0 for s in self.sensors]
        velocities = [
            finite_diff_velocity(positions[i], self.prev_positions[i], self.timestep * 1e-3)
            for i in range(len(self.sensors))
        ]
        self.prev_positions = positions[:]
        self.axis_monitor.update_all(positions, velocities)
        return positions, velocities

    def run(self):
        """Executes the master task-space state machine loop.

        Contracts:
            Invariants:
                - Loop runs iteratively until simulation kernel signs off (-1).
                - While self.state_counter > 0, system locks out computations
                  and decrements.
                - While state is VERTICAL_DESCEND, task-space XY coordinates
                  cannot change.
                - Task space trajectory Z depth cannot pass self.MIN_Z_SAFETY_LIMIT.
        """
        step_idx = 0
        while self.robot.step(self.timestep) != -1:
            sim_time = self.robot.getTime()
            q_current, q_dot_current = self.update_hardware_readings()

            if self.state_counter > 0:
                self.state_counter -= 1
                continue

            # State machine tracking execution logic
            if self.state == "INITIALIZE":
                init_pose = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
                for motor, q_target in zip(self.motors, init_pose):
                    motor.setVelocity(0.4)
                    motor.setPosition(q_target)

                max_error = max([abs(q_current[i] - init_pose[i]) for i in range(6)])
                if max_error < 0.005:
                    self.state = "CAMERA_TRACK_ALIGN"
                    self.state_counter = 10
                    print(f"[{sim_time:.2f}] Initialized. Advancing to CAMERA_TRACK_ALIGN.")

            elif self.state == "CAMERA_TRACK_ALIGN":
                blob = self.process_hand_camera_blobs()
                if not blob:
                    if step_idx % 40 == 0:
                        print(
                            f"[{sim_time:.2f}] Tracking Fault: Waiting for visual color signature matching '{self.grasp_target}'")
                    continue

                bx, by, pixel_count = blob
                w = self.camera.getWidth()
                h = self.camera.getHeight()

                error_x = (bx - (w / 2.0)) / (w / 2.0)
                error_y = ((h / 2.0) - by) / (h / 2.0)

                if abs(error_x) < 0.03 and abs(error_y) < 0.03:
                    frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
                    X_WG = self.plant.CalcRelativeTransform(self.context, self.plant.world_frame(), frame_palm)
                    current_xyz = X_WG.translation()

                    self.locked_target_xy = (current_xyz[0] + self.ARM_BASE_X, current_xyz[1] + self.ARM_BASE_Y)
                    self.current_descend_z = 0.45

                    self.state = "VERTICAL_DESCEND"
                    self.state_counter = 5
                    print(
                        f"[{sim_time:.2f}] Target centered cleanly. Horizontal coordinates locked. Proceeding to VERTICAL_DESCEND.")
                else:
                    kp = 0.035
                    dx = error_y * kp
                    dy = -error_x * kp

                    frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
                    X_WG = self.plant.CalcRelativeTransform(self.context, self.plant.world_frame(), frame_palm)
                    current_xyz = X_WG.translation()

                    target_x = current_xyz[0] + self.ARM_BASE_X + dx
                    target_y = current_xyz[1] + self.ARM_BASE_Y + dy

                    xyz_target = np.array([target_x, target_y, 0.45])
                    rpy_target = np.array([math.pi, 0.0, 0.0])

                    q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                    if q_opt:
                        for motor, q_t in zip(self.motors, q_opt):
                            motor.setVelocity(0.15)
                            motor.setPosition(q_t)

            elif self.state == "VERTICAL_DESCEND":
                dist = self.read_range_distance()

                if dist is not None and dist <= self.GRASP_DISTANCE_THRESHOLD:
                    self.state = "CLOSE_GRIPPER"
                    self.state_counter = 55
                    print(f"[{sim_time:.2f}] TRIGGER: Sensor read {dist:.4f}m. Envelope complete. Closing jaws.")

                    for gm in self.gripper_motors:
                        gm.setVelocity(0.2)
                        gm.setPosition(0.85)

                elif self.current_descend_z <= self.MIN_Z_SAFETY_LIMIT:
                    self.state = "IDLE"
                    print(
                        f"[{sim_time:.2f}] HARD SAFETY INTERCEPT: Floor threshold crossed without sensor lock. Halting.")

                else:
                    descent_speed = 0.0015 if (dist is not None and dist > 0.10) else 0.0008
                    self.current_descend_z -= descent_speed

                    xyz_target = np.array([self.locked_target_xy[0], self.locked_target_xy[1], self.current_descend_z])
                    rpy_target = np.array([math.pi, 0.0, 0.0])

                    q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                    if q_opt:
                        for motor, q_t in zip(self.motors, q_opt):
                            motor.setVelocity(0.1)
                            motor.setPosition(q_t)

            elif self.state == "CLOSE_GRIPPER":
                print(f"[{sim_time:.2f}] Grip sequence verified. Target manipulation secured.")
                self.state = "IDLE"

            elif self.state == "IDLE":
                pass

            if step_idx % 50 == 0:
                self.axis_monitor.print_table(sim_time, step_idx)
            step_idx += 1


if __name__ == "__main__":
    controller = RobotController()
    controller.run()