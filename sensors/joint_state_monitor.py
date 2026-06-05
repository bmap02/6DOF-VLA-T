"""sensors/joint_state_monitor.py — per-joint statistics for UR10e."""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from utils.kinematics_utils import rolling_mean, rolling_std, clamp

HISTORY_LEN   = 200
VELOCITY_EPS  = 1e-4

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

JOINT_LIMITS = [
    (-6.28318, 6.28318),
    (-6.28318, 6.28318),
    (-3.14159, 3.14159),
    (-6.28318, 6.28318),
    (-6.28318, 6.28318),
    (-6.28318, 6.28318),
]


@dataclass
class AxisStats:
    name:   str
    limits: tuple[float, float]

    pos_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    vel_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    max_velocity:   float = 0.0
    cum_motion:     float = 0.0
    total_samples:  int   = 0
    vel_sum:        float = 0.0

    prev_position: Optional[float] = None
    prev_velocity: Optional[float] = None

    def update(self, position: float, velocity: float) -> None:
        if self.prev_position is not None:
            self.cum_motion += abs(position - self.prev_position)
        self.pos_history.append(position)
        self.vel_history.append(velocity)
        if abs(velocity) > self.max_velocity:
            self.max_velocity = abs(velocity)
        self.vel_sum       += abs(velocity)
        self.total_samples += 1
        self.prev_position  = position
        self.prev_velocity  = velocity

    def reset_stats(self) -> None:
        self.max_velocity  = 0.0
        self.cum_motion    = 0.0
        self.total_samples = 0
        self.vel_sum       = 0.0
        self.pos_history.clear()
        self.vel_history.clear()
        self.prev_position = None
        self.prev_velocity = None

    @property
    def mean_speed(self) -> float:
        return self.vel_sum / self.total_samples if self.total_samples else 0.0

    @property
    def rolling_mean_speed(self) -> float:
        return rolling_mean(list(self.vel_history))

    @property
    def rolling_std_speed(self) -> float:
        return rolling_std(list(self.vel_history))

    @property
    def range_used(self) -> float:
        lo, hi = self.limits
        total = hi - lo
        if total == 0 or not self.pos_history:
            return 0.0
        return (max(self.pos_history) - min(self.pos_history)) / total

    @property
    def is_active(self) -> bool:
        return abs(self.prev_velocity or 0.0) > VELOCITY_EPS

    @property
    def near_limit(self) -> bool:
        if self.prev_position is None:
            return False
        lo, hi  = self.limits
        margin  = (hi - lo) * 0.05
        return self.prev_position < lo + margin or self.prev_position > hi - margin

    def to_dict(self, sim_time: float) -> dict:
        return {
            "t":             round(sim_time, 4),
            "joint":         self.name,
            "position":      round(self.prev_position or 0.0, 6),
            "velocity":      round(self.prev_velocity or 0.0, 6),
            "max_velocity":  round(self.max_velocity, 6),
            "mean_speed":    round(self.mean_speed, 6),
            "roll_mean_spd": round(self.rolling_mean_speed, 6),
            "roll_std_spd":  round(self.rolling_std_speed, 6),
            "cum_motion":    round(self.cum_motion, 6),
            "range_used":    round(self.range_used, 4),
            "is_active":     int(self.is_active),
            "near_limit":    int(self.near_limit),
        }


class AxisMonitor:
    def __init__(self) -> None:
        self.axes = [
            AxisStats(name=JOINT_NAMES[i], limits=JOINT_LIMITS[i])
            for i in range(6)
        ]

    def update_all(self, positions: list[float], velocities: list[float]) -> None:
        for ax, pos, vel in zip(self.axes, positions, velocities):
            ax.update(pos, vel)

    def snapshot(self, sim_time: float) -> list[dict]:
        return [ax.to_dict(sim_time) for ax in self.axes]

    def print_table(self, sim_time: float, step: int) -> None:
        C = {"b": "\033[1;34m", "g": "\033[32m", "d": "\033[90m",
             "y": "\033[33m",   "r": "\033[0m"}
        ROWS = 9  # header + column line + divider + 6 joints
        # After first print, jump cursor back up to overwrite
        if step > 0:
            print(f"\033[{ROWS}A", end="")
        lines = []
        lines.append(f"{C['b']}--- UR10e Joint Monitor{C['r']}  t={sim_time:.3f}s  step={step:<6}")
        lines.append(f"{'Joint':<18} {'pos':>8} {'vel':>8} {'peak':>7} {'avg':>7} {'cumul':>8}  {'range%':>7}  status")
        lines.append("-" * 74)
        for ax in self.axes:
            pos = ax.prev_position or 0.0
            vel = ax.prev_velocity or 0.0
            if ax.near_limit:
                flag = f"{C['y']}! limit{C['r']}"
            elif ax.is_active:
                flag = f"{C['g']}> move{C['r']}"
            else:
                flag = f"{C['d']}o idle{C['r']}"
            lines.append(
                f"{ax.name:<18} {pos:>+8.4f} {vel:>+8.4f} "
                f"{ax.max_velocity:>7.4f} {ax.mean_speed:>7.4f} "
                f"{ax.cum_motion:>8.4f}  {ax.range_used*100:>6.1f}%  {flag}"
            )
        # Print all lines, clearing to end of line on each
        for line in lines:
            print(line + "\033[K")
        # flush so Webots console shows update immediately
        import sys as _sys
        _sys.stdout.flush()

    def reset_all(self) -> None:
        for ax in self.axes:
            ax.reset_stats()