# overhead_camera_detection.py - Fixed overhead camera controller
#
# Scans the table for colored objects, converts pixel coordinates to
# world (x, y) coordinates using known camera geometry, and writes
# results to a shared JSON file that ur10e_touchnet.py reads each step.
#
# Camera setup (must match world file):
#   translation -5.8 -0.07 2.0
#   rotation 0 1 0 1.5708   (pointing straight down, -Z world direction)
#   fieldOfView default ~0.785 rad (45 deg) unless overridden in .wbt

import sys
import os
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from controller import Robot, Camera

TIMESTEP = 32

# Camera world position/orientation -- MUST match the world file exactly.
CAM_X = -5.8
CAM_Y = -0.07
CAM_Z = 2.0
TABLE_Z = 0.81   # object resting height, used to compute ground-plane distance

# Target colors -- same definitions as main controller, kept in sync manually.
TARGET_COLORS = {
    "red":    {"rgb": (255, 0,   0),   "tol": 70},
    "blue":   {"rgb": (85,  170, 255), "tol": 70},
    "yellow": {"rgb": (255, 255, 0),   "tol": 70},
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "overhead_detections.json")


def detect_blobs(camera):
    MIN_PIXELS = 10
    SAMPLE_STEP = 3

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


def pixel_to_world(px, py, width, height, fov_h, cam_x, cam_y, cam_z, ground_z):
    """
    Convert pixel coordinates to world (x, y) on the ground plane,
    assuming camera looks straight down (-Z) with horizontal FOV fov_h.

    Pixel (width/2, height/2) maps to (cam_x, cam_y) directly below camera.
    """
    height_above_ground = cam_z - ground_z
    fov_v = fov_h * (height / width)  # approximate vertical FOV from aspect ratio

    # Half-extent of the visible ground patch at this height
    half_width_world  = height_above_ground * math.tan(fov_h / 2)
    half_height_world = height_above_ground * math.tan(fov_v / 2)

    # Normalized offset from image center, range [-1, 1]
    norm_x = (px - width / 2) / (width / 2)
    norm_y = (py - height / 2) / (height / 2)

    # Camera is rotated 0 1 0 1.5708 -- this maps image x -> world y, image y -> world x
    # (rotation about Y axis by 90 deg swaps the horizontal axes)
    world_x = cam_x - norm_y * half_height_world
    world_y = cam_y + norm_x * half_width_world

    return world_x, world_y


def main():
    robot = Robot()

    camera = robot.getDevice("overhead_camera")
    if camera is None:
        print("[overhead_camera_detection] ERROR: overhead_camera device not found.")
        return
    camera.enable(TIMESTEP)
    print("[overhead_camera_detection] Overhead camera enabled.")

    width  = camera.getWidth()
    height = camera.getHeight()
    fov_h  = camera.getFov()

    print(f"[overhead_camera_detection] Resolution {width}x{height}, FOV {fov_h:.3f} rad")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    step = 0
    while robot.step(TIMESTEP) != -1:
        blobs = detect_blobs(camera)

        detections = {}
        for name, (px, py, count) in blobs.items():
            wx, wy = pixel_to_world(px, py, width, height, fov_h,
                                     CAM_X, CAM_Y, CAM_Z, TABLE_Z)
            detections[name] = {
                "world_x": round(wx, 4),
                "world_y": round(wy, 4),
                "pixel_x": round(px, 1),
                "pixel_y": round(py, 1),
                "pixel_count": count,
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