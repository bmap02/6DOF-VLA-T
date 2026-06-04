"""pipeline/episode_runner.py — runs a single grasp episode end-to-end.

Ties together the robot controller, sensors, contact model, and logger.
The VLAInterface stub is where the VLA model plugs in later.
"""

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from objects.object_library import GraspObject
from physics.contact_model import HertzContact
from sensors.joint_state_monitor import AxisMonitor
from sensors.tactile_array_sensor import TactileSensor
from sensors.ft_sensor_interface import ForceTorqueSensor
from pipeline.log_export import DataLogger


# ── VLA interface (implement this when your model is ready) ───────────────────

class VLAInterface(ABC):
    """
    Abstract interface for a Vision-Language-Action model controlling
    gripper force. Subclass this and pass an instance to EpisodeRunner.

    Input
    -----
    tactile_map : (16, 16) float32 — taxel pressures in Pa
    joint_state : list[float] — 6 joint positions in rad
    ft_wrench   : (fx, fy, fz, tx, ty, tz) in N / N·m
    object_desc : str — natural language description of the object

    Output
    ------
    float — target gripper closing force in N (0 = open, max ~50 N)
    """

    @abstractmethod
    def predict_grasp_force(
        self,
        tactile_map:  "np.ndarray",
        joint_state:  list[float],
        ft_wrench:    tuple[float, ...],
        object_desc:  str = "",
    ) -> float:
        ...


class ConstantForcePolicy(VLAInterface):
    """Baseline: always apply the same closing force. Good for benchmarking."""

    def __init__(self, force_n: float = 5.0) -> None:
        self.force = force_n

    def predict_grasp_force(self, tactile_map, joint_state, ft_wrench, object_desc="") -> float:
        return self.force


class ProportionalForcePolicy(VLAInterface):
    """Baseline: scale force proportionally to current grip margin."""

    def __init__(self, target_margin: float = 0.3, k: float = 10.0) -> None:
        self.target_margin = target_margin
        self.k = k

    def predict_grasp_force(self, tactile_map, joint_state, ft_wrench, object_desc="") -> float:
        # Simple P-controller on grip margin (read from last contact state externally)
        # Placeholder — real impl reads from shared state
        return 5.0


# ── Episode config ────────────────────────────────────────────────────────────

@dataclass
class EpisodeConfig:
    scenario:       str              # 'benchmark_stable_grasp' | 'slip_detection' | ...
    object_name:    str
    max_steps:      int   = 500
    timestep_ms:    int   = 32
    policy:         VLAInterface | None = None
    log_taxel_maps: bool  = False    # set True to log full 16×16 arrays


# ── Episode runner ────────────────────────────────────────────────────────────

class EpisodeRunner:
    """
    Orchestrates one grasp episode.

    In Webots, instantiate this inside the robot controller and call
    step() each simulation timestep. Outside Webots, call run_offline()
    with pre-recorded joint data for post-processing.
    """

    def __init__(
        self,
        config: EpisodeConfig,
        robot=None,   # Webots Robot instance (None for offline use)
    ) -> None:
        self.cfg     = config
        self.robot   = robot

        self.joint_state_monitor    = AxisMonitor()
        self.tactile_left    = TactileSensor(noise_std=50.0)
        self.tactile_right   = TactileSensor(noise_std=50.0)
        self.ft_sensor       = ForceTorqueSensor()
        self.contact_model   = HertzContact(finger_radius=0.010)
        self.logger          = DataLogger()
        self.policy          = config.policy or ConstantForcePolicy(force_n=5.0)

        self._episode_id     = f"ep_{uuid.uuid4().hex[:8]}"
        self._step           = 0
        self._sim_time       = 0.0
        self._dt             = config.timestep_ms * 1e-3
        self._prev_positions = [None] * 6

    def start(self, obj: GraspObject) -> None:
        self._obj = obj
        self.logger.start(
            episode_id  = self._episode_id,
            object_name = obj.name,
            scenario    = self.cfg.scenario,
            extra_meta  = {"material": obj.material.name, "category": obj.material.category},
        )
        self.ft_sensor.tare()
        print(f"[EpisodeRunner] Starting {self._episode_id} — {self.cfg.scenario} / {obj.name}")

    def step(
        self,
        positions:         list[float],   # 6 joint positions (rad)
        velocities:        list[float],   # 6 joint velocities (rad/s)
        gripper_force:     float,         # N — current gripper force
        tangential_force:  float = 0.0,   # N — estimated slip force
        slip_velocity:     float = 0.0,   # m/s
        contact_x:        float = 0.0,   # m
        contact_y:        float = 0.0,   # m
    ) -> float:
        """
        Process one timestep. Returns recommended gripper force from policy.
        """
        self._sim_time = self._step * self._dt
        self._step    += 1

        # ── Axis monitor ──────────────────────────────────────────────────
        self.joint_state_monitor.update_all(positions, velocities)
        axes_dicts = self.joint_state_monitor.snapshot(self._sim_time)

        # ── Contact model ─────────────────────────────────────────────────
        contact = self.contact_model.compute(
            normal_force     = gripper_force,
            tangential_force = tangential_force,
            slip_velocity    = slip_velocity,
            mat_obj          = self._obj.material,
        )

        # ── Tactile sensors ───────────────────────────────────────────────
        t_left  = self.tactile_left.compute(contact,  contact_x,  contact_y)
        t_right = self.tactile_right.compute(contact, -contact_x, contact_y)

        # ── F/T sensor ────────────────────────────────────────────────────
        ft = self.ft_sensor.read(
            true_wrench=(gripper_force, tangential_force, 0.0, 0.0, 0.0, 0.0),
            timestamp  = self._sim_time,
        )

        # ── VLA policy ────────────────────────────────────────────────────
        recommended_force = self.policy.predict_grasp_force(
            tactile_map = t_left.pressure_map,
            joint_state = positions,
            ft_wrench   = (ft.fx, ft.fy, ft.fz, ft.tx, ft.ty, ft.tz),
            object_desc = f"{self._obj.material.category} object: {self._obj.name}",
        )

        # ── Log ───────────────────────────────────────────────────────────
        tactile_dict = t_left.to_dict()
        if not self.cfg.log_taxel_maps:
            tactile_dict.pop("taxel_pressures", None)

        self.logger.log(
            sim_time    = self._sim_time,
            axes_dicts  = axes_dicts,
            contact     = contact.to_dict(),
            tactile     = tactile_dict,
            ft          = ft.to_dict(),
        )

        return recommended_force

    def stop(self, outcome: str = "unknown", metrics: dict | None = None):
        self.logger.stop(outcome=outcome, metrics=metrics)
        print(f"[EpisodeRunner] Episode {self._episode_id} finished — {outcome}")