"""benchmarks/benchmark_stable_grasp.py — benchmark: stable grasping.

Runs N episodes across rigid and soft objects, measures grip stability
(no slip events, force within safe range) and logs outcomes.

Run offline (no Webots) with simulated joint/force data for fast iteration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from objects.object_library import by_category, GraspObject
from pipeline.episode_runner import EpisodeConfig, EpisodeRunner, ConstantForcePolicy
from utils.kinematics_utils import clamp


FORCE_LEVELS_N = [2.0, 4.0, 6.0, 8.0, 12.0]
N_STEPS        = 200


def _fake_positions(step: int) -> list[float]:
    """Simulate a slow approach trajectory for offline testing."""
    t = step * 0.032
    return [
        0.3 * np.sin(0.5 * t),
        -0.2 + 0.1 * np.sin(0.3 * t),
        0.5 + 0.05 * np.sin(0.4 * t),
        0.0,
        0.5 * np.sin(0.2 * t),
        0.0,
    ]


def run_episode(obj: GraspObject, force_n: float) -> dict:
    config = EpisodeConfig(
        scenario    = "benchmark_stable_grasp",
        object_name = obj.name,
        max_steps   = N_STEPS,
        policy      = ConstantForcePolicy(force_n=force_n),
    )
    runner = EpisodeRunner(config=config)
    runner.start(obj)

    slip_events       = 0
    force_exceeded    = False
    prev_positions    = None

    for step in range(N_STEPS):
        positions = _fake_positions(step)
        if prev_positions is None:
            velocities = [0.0] * 6
        else:
            velocities = [(positions[i] - prev_positions[i]) / 0.032 for i in range(6)]
        prev_positions = positions[:]

        tangential = force_n * 0.15 * (1 + 0.1 * np.random.randn())

        recommended = runner.step(
            positions       = positions,
            velocities      = velocities,
            gripper_force   = force_n,
            tangential_force= tangential,
            slip_velocity   = 0.01 * max(0, tangential - obj.material.mu_s * force_n),
        )

        # Check outcomes
        from physics.contact_model import HertzContact
        contact = HertzContact().compute(force_n, tangential, 0.0, obj.material)
        if contact.is_slipping:
            slip_events += 1
        if force_n > obj.max_safe_force:
            force_exceeded = True

    success = slip_events == 0 and not force_exceeded
    metrics = {
        "force_n":       force_n,
        "slip_events":   slip_events,
        "force_exceeded": int(force_exceeded),
        "success":       int(success),
    }
    runner.stop(outcome="success" if success else "failure", metrics=metrics)
    return metrics


def main():
    print("=== Stable Grasp Benchmark ===\n")
    results = []

    test_objects = by_category("rigid")[:3] + by_category("soft")[:3]

    for obj in test_objects:
        for force in FORCE_LEVELS_N:
            m = run_episode(obj, force)
            results.append({"object": obj.name, **m})
            tag = "PASS" if m["success"] else "FAIL"
            print(f"  [{tag}] {obj.name:<20} force={force:.1f}N  slip={m['slip_events']}")

    total   = len(results)
    passed  = sum(1 for r in results if r["success"])
    print(f"\nResult: {passed}/{total} passed  ({100*passed/total:.0f}%)")


if __name__ == "__main__":
    main()