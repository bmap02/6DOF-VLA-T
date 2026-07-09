import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import math
import numpy as np
from controller import Robot

# --- Drake Kinematics & Optimization Engine ---
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame
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
            self.camera.recognitionEnable(self.timestep)

        # Webots Base Anchor Frame Transformation Offsets
        # (must match the UR10e node's translation/rotation in the .wbt world file)
        self.ARM_BASE_X = -4.84
        self.ARM_BASE_Y = -0.35
        self.ARM_BASE_Z = 0.742
        self.ARM_BASE_ROT = 1.5708

        # Coarse Approach Target -- center of the "ObjectBoundingBox" region
        # marker in the .wbt world file (translation, and size/2 for the
        # half-extent), hovering above the workspace so the narrow-FOV wrist
        # camera has something to search over. Any camera detection outside
        # this region is treated as background noise, not a pick candidate --
        # this is what makes "the object to pick" dynamic (whatever is in
        # the box) rather than a hardcoded position or color.
        self.OBJECT_REGION_CENTER_X = -5.79
        self.OBJECT_REGION_CENTER_Y = -0.05
        self.OBJECT_REGION_HALF_EXTENT_XY = 0.25
        self.APPROACH_HOVER_Z = 1.2
        # Height used for horizontal transit moves -- comfortably above the
        # workspace bounding box (which tops out at world Z=1.31) so the arm
        # never has to cut across near table height to get there.
        self.SAFE_TRANSIT_Z = 1.4

        # Shared-file handoff from the overhead camera controller (see
        # controllers/overhead_camera_detection/overhead_camera_detection.py)
        self.OVERHEAD_DETECTIONS_PATH = os.path.join(
            os.path.dirname(__file__), "..", "..", "logs", "overhead_detections.json"
        )

        # 3. Instantiate In-Memory Drake Kinematic Engine
        self.plant, self.arm_model, self.gripper_model, self.wrist_camera_frame = self._build_drake_engine()
        self.context = self.plant.CreateDefaultContext()

        # 4. Initialize Core Monitors & Models
        self.axis_monitor = AxisMonitor()
        self.contact_model = HertzContact(finger_radius=0.010)

        # Operational State Machine Variables
        self.state = "INITIALIZE"

        self.prev_positions = [None] * len(self.sensors)
        self.current_descend_z = self.APPROACH_HOVER_Z
        self.locked_target_xy = None
        # World Z of the tracked object's center (from Recognition), captured
        # when CAMERA_TRACK_ALIGN locks. The gripper's fingertips sit close
        # to the palm's own height (measured via forward kinematics, not
        # hanging far below it), so this -- not a blind rangefinder distance
        # -- is the reliable signal for how far the palm actually needs to
        # descend to reach the object.
        self.grasp_target_z = None
        # Overhead-camera region estimate, snapshotted once before the arm
        # sweeps horizontally into the workspace (see TRANSIT_UP). Once the
        # arm's own body enters the overhead camera's view it can itself get
        # picked up as a "blob", so this must not be re-queried every step.
        self.locked_region_xy = None
        # Webots Recognition node id of the object locked onto by the
        # overhead camera (see TRANSIT_UP). The wrist camera looks up this
        # exact id in its own recognition list during CAMERA_TRACK_ALIGN --
        # an unambiguous identity match, unlike picking "largest blob" each
        # frame, which can flip between two similarly-sized objects.
        self.tracked_object_id = None
        self.calculated_yaw = 0.0

        # Execution Guard Variables
        self.state_counter = 0
        # World Z hard floor for descent -- just above the table surface
        # (objects rest at world Z=0.81; table top ~0.74) so a missed
        # rangefinder trigger halts before the palm drives into the table.
        self.MIN_Z_SAFETY_LIMIT = 0.75

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

        # Wrist camera mount offset relative to tool0 -- must match the
        # wrist_camera Camera node's translation/rotation under toolSlot in
        # the .wbt world file. Lets us convert Webots Recognition object
        # positions (given relative to the camera) into world coordinates
        # via ordinary forward kinematics.
        X_ToolCam = RigidTransform(
            RotationMatrix(AngleAxis(
                3.14159, [0.70710528118436, 0.707108281185553, 3.3905113482557537e-09]
            )),
            [-1.93319e-07, 0.1, -1.01284e-07],
        )
        wrist_camera_frame = plant.AddFrame(FixedOffsetFrame("wrist_camera", frame_tool0, X_ToolCam))

        # Anchor base_link to the world frame at the UR10e node's actual
        # Webots pose. Without this weld, base_link is left unconnected to
        # the world and Drake silently adds a 6-DOF floating joint for it --
        # the IK solution vector then starts with that floating body's
        # quaternion/xyz instead of the arm's joint angles.
        frame_base = plant.GetFrameByName("base_link", arm_model)
        X_WBase = RigidTransform(
            RotationMatrix(RollPitchYaw(0.0, 0.0, self.ARM_BASE_ROT)),
            [self.ARM_BASE_X, self.ARM_BASE_Y, self.ARM_BASE_Z],
        )
        plant.WeldFrames(plant.world_frame(), frame_base, X_WBase)

        plant.Finalize()
        return plant, arm_model, gripper_model, wrist_camera_frame

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
        """
        ik = InverseKinematics(self.plant, self.context)
        frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
        frame_world = self.plant.world_frame()

        target_xyz_world = np.asarray(target_xyz_world, dtype=float)

        ik.AddPositionConstraint(
            frame_palm, np.zeros(3),
            frame_world, target_xyz_world - 0.002, target_xyz_world + 0.002
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

    def locate_tracked_object_world_xyz(self):
        """Finds self.tracked_object_id's world (x, y, z) via the wrist camera.

        Uses Webots Camera Recognition (ground-truth position relative to
        the camera, https://cyberbotics.com/doc/reference/recognition)
        rather than color-blob pixel detection, and matches by the object's
        stable Recognition id rather than proximity -- this is what prevents
        the arm from flip-flopping onto a different, similarly-placed object.

        Returns:
            tuple(float, float, float) or None: World (x, y, z) of the
                tracked object's center; None if it isn't currently
                visible/recognized, or no object is being tracked yet.

        Contracts:
            Pre-conditions:
                - Camera hardware device must be active with recognition
                  enabled.
        """
        if not self.camera or self.tracked_object_id is None:
            return None
        for obj in self.camera.getRecognitionObjects():
            if obj.getId() != self.tracked_object_id:
                continue
            X_WCam = self.plant.CalcRelativeTransform(
                self.context, self.plant.world_frame(), self.wrist_camera_frame
            )
            rel_pos = np.array(obj.getPosition())
            world_pos = X_WCam.translation() + X_WCam.rotation().matrix() @ rel_pos
            return (world_pos[0], world_pos[1], world_pos[2])
        return None

    def read_overhead_detection(self):
        """Reads the overhead camera's most recent object detection.

        Returns:
            dict or None: {"id", "model", "world_x", "world_y", "world_z"}
                of the last detected object; None if the file is missing,
                malformed, or no object has been detected yet.

        Contracts:
            Pre-conditions:
                - controllers/overhead_camera_detection is running and
                  writing to self.OVERHEAD_DETECTIONS_PATH.
        """
        try:
            with open(self.OVERHEAD_DETECTIONS_PATH, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data.get("detections", {}).get("object")

    def resolve_object_region(self):
        """Best-known estimate of the object to approach: world (x, y) + id.

        Prefers the overhead camera's live detection, but only if it falls
        within the ObjectBoundingBox region -- this is what makes "the
        object" dynamic (whatever is physically inside the pick region)
        rather than a hardcoded position or color. Detections outside the
        region are background/noise and are discarded. Falls back to the
        region's center (with no id to track) if no valid detection has
        been written yet.

        Returns:
            tuple(float, float, int or None): (world_x, world_y, id).
        """
        det = self.read_overhead_detection()
        if det is not None:
            dx, dy = det["world_x"], det["world_y"]
            if (abs(dx - self.OBJECT_REGION_CENTER_X) <= self.OBJECT_REGION_HALF_EXTENT_XY
                    and abs(dy - self.OBJECT_REGION_CENTER_Y) <= self.OBJECT_REGION_HALF_EXTENT_XY):
                return (dx, dy, det.get("id"))
        return (self.OBJECT_REGION_CENTER_X, self.OBJECT_REGION_CENTER_Y, None)

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
                - Syncs self.context with the arm's live joint positions so
                  Drake queries (CalcRelativeTransform, compute_ik seeding)
                  reflect the robot's actual current pose.
        """
        positions = [s.getValue() if s else 0.0 for s in self.sensors]
        velocities = [
            finite_diff_velocity(positions[i], self.prev_positions[i], self.timestep * 1e-3)
            for i in range(len(self.sensors))
        ]
        self.prev_positions = positions[:]
        self.axis_monitor.update_all(positions, velocities)

        q_full = np.array(self.plant.GetPositions(self.context))
        q_full[:len(positions)] = positions
        self.plant.SetPositions(self.context, q_full)

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
                    self.state = "TRANSIT_UP"
                    self.state_counter = 10
                    print(f"[{sim_time:.2f}] Initialized. Advancing to TRANSIT_UP.")

            elif self.state == "TRANSIT_UP":
                # Rise straight up in place before moving horizontally, so
                # the arm never cuts a diagonal path down toward the table.
                frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
                X_WG = self.plant.CalcRelativeTransform(self.context, self.plant.world_frame(), frame_palm)
                current_xyz = X_WG.translation()

                xyz_target = np.array([current_xyz[0], current_xyz[1], self.SAFE_TRANSIT_Z])
                rpy_target = np.array([math.pi, 0.0, 0.0])

                q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                if q_opt:
                    for motor, q_t in zip(self.motors, q_opt):
                        motor.setVelocity(0.3)
                        motor.setPosition(q_t)

                    max_error = max([abs(q_current[i] - q_opt[i]) for i in range(6)])
                    if max_error < 0.01:
                        # Snapshot the overhead-camera region estimate now,
                        # while the arm is still directly above its own base
                        # and hasn't entered the workspace yet. Once it does
                        # (TRANSIT_OVER/MOVE_TO_REGION), the arm itself can
                        # show up as a recognized "object" to the overhead
                        # camera, so this must not be re-queried afterward.
                        region_x, region_y, self.tracked_object_id = self.resolve_object_region()
                        self.locked_region_xy = (region_x, region_y)
                        self.state = "TRANSIT_OVER"
                        self.state_counter = 10
                        print(f"[{sim_time:.2f}] Reached transit height. Region locked at "
                              f"({self.locked_region_xy[0]:.3f}, {self.locked_region_xy[1]:.3f}), "
                              f"tracking object id={self.tracked_object_id}. Advancing to TRANSIT_OVER.")
                else:
                    print(f"[{sim_time:.2f}] TRANSIT_UP: IK solve failed.")

            elif self.state == "TRANSIT_OVER":
                region_x, region_y = self.locked_region_xy
                xyz_target = np.array([region_x, region_y, self.SAFE_TRANSIT_Z])
                rpy_target = np.array([math.pi, 0.0, 0.0])

                q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                if q_opt:
                    for motor, q_t in zip(self.motors, q_opt):
                        motor.setVelocity(0.3)
                        motor.setPosition(q_t)

                    max_error = max([abs(q_current[i] - q_opt[i]) for i in range(6)])
                    if max_error < 0.01:
                        self.state = "MOVE_TO_REGION"
                        self.state_counter = 10
                        print(f"[{sim_time:.2f}] Positioned over object region. Advancing to MOVE_TO_REGION.")
                else:
                    print(f"[{sim_time:.2f}] TRANSIT_OVER: IK solve failed for region target.")

            elif self.state == "MOVE_TO_REGION":
                region_x, region_y = self.locked_region_xy
                xyz_target = np.array([region_x, region_y, self.APPROACH_HOVER_Z])
                rpy_target = np.array([math.pi, 0.0, 0.0])

                q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                if q_opt:
                    for motor, q_t in zip(self.motors, q_opt):
                        motor.setVelocity(0.2)
                        motor.setPosition(q_t)

                    max_error = max([abs(q_current[i] - q_opt[i]) for i in range(6)])
                    if max_error < 0.01:
                        self.state = "CAMERA_TRACK_ALIGN"
                        self.state_counter = 10
                        print(f"[{sim_time:.2f}] Reached object region. Advancing to CAMERA_TRACK_ALIGN.")
                else:
                    print(f"[{sim_time:.2f}] MOVE_TO_REGION: IK solve failed for region target.")

            elif self.state == "CAMERA_TRACK_ALIGN":
                target_xyz = self.locate_tracked_object_world_xyz()
                if target_xyz is None:
                    if step_idx % 40 == 0:
                        print(f"[{sim_time:.2f}] Tracking Fault: Tracked object (id={self.tracked_object_id}) "
                              f"not visible in wrist camera view.")
                    step_idx += 1
                    continue

                frame_palm = self.plant.GetFrameByName("palm", self.gripper_model)
                X_WG = self.plant.CalcRelativeTransform(self.context, self.plant.world_frame(), frame_palm)
                current_xyz = X_WG.translation()

                error_x = target_xyz[0] - current_xyz[0]
                error_y = target_xyz[1] - current_xyz[1]

                if abs(error_x) < 0.005 and abs(error_y) < 0.005:
                    self.locked_target_xy = (target_xyz[0], target_xyz[1])
                    # The tracked object's own recognized height -- the
                    # reliable target for how far down the palm needs to go
                    # (see the state_counter=270 comment in VERTICAL_DESCEND).
                    self.grasp_target_z = target_xyz[2]
                    self.current_descend_z = self.APPROACH_HOVER_Z

                    self.state = "VERTICAL_DESCEND"
                    self.state_counter = 5
                    print(
                        f"[{sim_time:.2f}] Target centered cleanly. Horizontal coordinates locked "
                        f"(grasp Z={self.grasp_target_z:.3f}). Proceeding to VERTICAL_DESCEND.")
                else:
                    xyz_target = np.array([target_xyz[0], target_xyz[1], self.APPROACH_HOVER_Z])
                    rpy_target = np.array([math.pi, 0.0, 0.0])

                    q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                    if q_opt:
                        for motor, q_t in zip(self.motors, q_opt):
                            motor.setVelocity(0.15)
                            motor.setPosition(q_t)

            elif self.state == "VERTICAL_DESCEND":
                dist = self.read_range_distance()
                # Primary trigger: the palm has reached the object's own
                # recognized height (ground truth from Recognition). The
                # rangefinder is kept as a secondary/earlier trigger in case
                # it fires first, but is not solely trusted -- the gripper's
                # fingertips sit close to the palm's own height (measured via
                # forward kinematics), not far below it, so a rangefinder
                # miscalibration was causing the gripper to close on empty
                # air while still well above the object.
                reached_object_height = self.current_descend_z <= self.grasp_target_z
                rangefinder_triggered = dist is not None and dist <= self.GRASP_DISTANCE_THRESHOLD

                if reached_object_height or rangefinder_triggered:
                    self.state = "CLOSE_GRIPPER"
                    # Gripper motors move at 0.2 rad/s toward a 0.85 rad
                    # target -- give them the full ~4.3s (270 steps @ 16ms)
                    # to either reach it or stall against the object before
                    # the arm starts lifting.
                    self.state_counter = 270
                    reason = "reached object height" if reached_object_height else f"sensor read {dist:.4f}m"
                    print(f"[{sim_time:.2f}] TRIGGER: {reason}. Envelope complete. Closing jaws.")

                    for gm in self.gripper_motors:
                        gm.setVelocity(0.2)
                        gm.setPosition(0.85)

                elif self.current_descend_z <= self.MIN_Z_SAFETY_LIMIT:
                    self.state = "IDLE"
                    print(
                        f"[{sim_time:.2f}] HARD SAFETY INTERCEPT: Floor threshold crossed without sensor lock. Halting.")

                else:
                    descent_speed = 0.0015 if (dist is not None and dist > 0.10) else 0.0008
                    candidate_z = self.current_descend_z - descent_speed

                    xyz_target = np.array([self.locked_target_xy[0], self.locked_target_xy[1], candidate_z])
                    rpy_target = np.array([math.pi, 0.0, 0.0])

                    q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                    if q_opt:
                        # Only commit the deeper target once IK actually confirms
                        # it's reachable -- otherwise current_descend_z would keep
                        # sinking every step even while the arm is stuck at its
                        # last valid pose, and the height-reached trigger above
                        # would fire on a depth the palm never physically reached.
                        self.current_descend_z = candidate_z
                        for motor, q_t in zip(self.motors, q_opt):
                            motor.setVelocity(0.1)
                            motor.setPosition(q_t)
                    else:
                        print(f"[{sim_time:.2f}] VERTICAL_DESCEND: IK solve failed at "
                              f"z={candidate_z:.4f}; holding at z={self.current_descend_z:.4f}.")

            elif self.state == "CLOSE_GRIPPER":
                print(f"[{sim_time:.2f}] Grip sequence complete. Lifting object.")
                self.state = "LIFT"

            elif self.state == "LIFT":
                xyz_target = np.array([self.locked_target_xy[0], self.locked_target_xy[1], self.SAFE_TRANSIT_Z])
                rpy_target = np.array([math.pi, 0.0, 0.0])

                q_opt = self.compute_ik(xyz_target, rpy_target, q_current)
                if q_opt:
                    for motor, q_t in zip(self.motors, q_opt):
                        motor.setVelocity(0.15)
                        motor.setPosition(q_t)

                    max_error = max([abs(q_current[i] - q_opt[i]) for i in range(6)])
                    if max_error < 0.01:
                        self.state = "IDLE"
                        print(f"[{sim_time:.2f}] Object lifted clear of the table. Pick-and-lift sequence complete.")
                else:
                    print(f"[{sim_time:.2f}] LIFT: IK solve failed.")

            elif self.state == "IDLE":
                pass

            if step_idx % 50 == 0:
                self.axis_monitor.print_table(sim_time, step_idx)
            step_idx += 1


if __name__ == "__main__":
    controller = RobotController()
    controller.run()