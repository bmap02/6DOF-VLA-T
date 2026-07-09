# overhead_camera_detection.py - Fixed overhead camera controller
#
# Uses Webots' Camera Recognition (https://cyberbotics.com/doc/reference/recognition)
# instead of color-blob pixel scanning. Recognition gives ground-truth 3D
# object positions (relative to the camera) and a stable per-object `id`,
# so there's no flat-ground-plane assumption and no color-bucket ambiguity.
# Only Solid nodes with a non-empty `recognitionColors` field are seen (the
# Box/Cylinder/Sphere objects in the world file); the pick-region marker and
# background scenery are not, by construction.
#
# Writes the best in-region detection to a shared JSON file that
# ur10e_touchnet.py reads each step.
#
# Camera setup (must match world file):
#   translation -5.8 -0.07 2.0
#   rotation 0 1 0 1.5708

import sys
import os
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from controller import Robot

TIMESTEP = 32

# Camera world pose -- MUST match the world file's Robot node exactly.
CAM_TRANSLATION = np.array([-5.8, -0.07, 2.0])
CAM_ROTATION_AXIS = (0.0, 1.0, 0.0)
CAM_ROTATION_ANGLE = 1.5708

# Pick region -- must match the "ObjectBoundingBox" marker in the .wbt world
# file (translation, and size/2 for the half-extent) and
# ur10e_touchnet.py's OBJECT_REGION_CENTER_X/Y / OBJECT_REGION_HALF_EXTENT_XY.
# A detected object only counts as "the object" if it's physically inside
# this region -- objects elsewhere on the table are ignored.
REGION_CENTER_X = -5.79
REGION_CENTER_Y = -0.05
REGION_HALF_EXTENT_XY = 0.25

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "overhead_detections.json")


def axis_angle_to_matrix(axis, angle):
    """Rodrigues' rotation formula -- VRML SFRotation (axis, angle) to a 3x3 matrix."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


CAM_ROTATION_MATRIX = axis_angle_to_matrix(CAM_ROTATION_AXIS, CAM_ROTATION_ANGLE)


def recognized_object_to_world_xyz(obj):
    """Converts a CameraRecognitionObject's camera-relative position to world coordinates."""
    rel_pos = np.array(obj.getPosition())
    world_pos = CAM_TRANSLATION + CAM_ROTATION_MATRIX @ rel_pos
    return world_pos


def main():
    robot = Robot()

    camera = robot.getDevice("overhead_camera")
    if camera is None:
        print("[overhead_camera_detection] ERROR: overhead_camera device not found.")
        return
    camera.enable(TIMESTEP)
    camera.recognitionEnable(TIMESTEP)
    print("[overhead_camera_detection] Overhead camera recognition enabled.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    step = 0
    while robot.step(TIMESTEP) != -1:
        candidates = []
        for obj in camera.getRecognitionObjects():
            wx, wy, wz = recognized_object_to_world_xyz(obj)
            if (abs(wx - REGION_CENTER_X) > REGION_HALF_EXTENT_XY
                    or abs(wy - REGION_CENTER_Y) > REGION_HALF_EXTENT_XY):
                continue  # outside the pick region -- not a candidate
            candidates.append((wx, wy, wz, obj))

        detections = {}
        if candidates:
            # Deterministic tie-break among multiple in-region objects: closest to region center.
            wx, wy, wz, obj = min(
                candidates,
                key=lambda c: (c[0] - REGION_CENTER_X) ** 2 + (c[1] - REGION_CENTER_Y) ** 2,
            )
            detections["object"] = {
                "id": obj.getId(),
                "model": obj.getModel(),
                "world_x": round(float(wx), 4),
                "world_y": round(float(wy), 4),
                "world_z": round(float(wz), 4),
            }

        # Write to shared file every step (cheap, small JSON)
        try:
            with open(OUTPUT_PATH, "w") as f:
                json.dump({"step": step, "detections": detections}, f)
        except Exception as e:
            print(f"[overhead_camera_detection] Write error: {e}")

        if step % 60 == 0:
            print(f"[overhead_camera_detection] step={step} detections={detections}")

        step += 1


if __name__ == "__main__":
    main()
