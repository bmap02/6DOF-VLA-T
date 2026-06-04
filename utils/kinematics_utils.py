"""utils/math_utils.py — shared vector and rotation helpers."""

import math
from typing import Sequence


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v[:3]))


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a[:3], b[:3]))


def finite_diff_velocity(curr: float, prev: float | None, dt: float) -> float:
    if prev is None or dt <= 0:
        return 0.0
    return (curr - prev) / dt


def rolling_mean(history: list[float]) -> float:
    return sum(history) / len(history) if history else 0.0


def rolling_std(history: list[float]) -> float:
    if len(history) < 2:
        return 0.0
    m = rolling_mean(history)
    return math.sqrt(sum((x - m) ** 2 for x in history) / len(history))


def rad_to_deg(r: float) -> float:
    return r * 180.0 / math.pi