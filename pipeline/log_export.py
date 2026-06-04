"""pipeline/log_export.py — CPIM / TouchNet format trajectory writer.

Writes one row per timestep containing joint state, contact state,
tactile reading summary, and F/T reading. Output is a flat CSV and
a parallel JSON metadata file per episode.
"""

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from sensors.joint_state_monitor import AxisStats
from physics.contact_model import ContactState, ContactNoState
from sensors.tactile_array_sensor import TactileReading
from sensors.ft_sensor_interface import FTReading


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class DataLogger:
    """
    Opens a new episode file on start(), writes rows, closes on stop().

    Usage
    -----
    logger = DataLogger()
    logger.start(episode_id="ep_001", object_name="egg", scenario="force_overshoot")
    logger.log(sim_time, axes, contact, tactile, ft)
    logger.stop(outcome="success", metrics={...})
    """

    def __init__(self, log_dir: Path = LOG_DIR) -> None:
        self.log_dir = log_dir
        self._csv_file   = None
        self._csv_writer = None
        self._meta       = {}
        self._episode_id = None
        self._row_count  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        episode_id: str,
        object_name: str,
        scenario: str,
        extra_meta: dict | None = None,
    ) -> None:
        self._episode_id = episode_id
        self._row_count  = 0

        csv_path = self.log_dir / f"{episode_id}.csv"
        self._csv_file = open(csv_path, "w", newline="")

        # Build fieldnames from a dummy row
        dummy = self._build_row(
            0.0,
            [{"joint": "A1", "position": 0.0, "velocity": 0.0,
              "max_velocity": 0.0, "mean_speed": 0.0, "cum_motion": 0.0,
              "roll_std_spd": 0.0, "range_used": 0.0, "is_active": 0, "near_limit": 0}] * 6,
            ContactNoState().to_dict(),
            {"total_force": 0.0, "contact_area": 0.0,
             "centroid_x": 0.0, "centroid_y": 0.0, "is_contact": 0},
            {"fx": 0.0, "fy": 0.0, "fz": 0.0,
             "tx": 0.0, "ty": 0.0, "tz": 0.0,
             "force_mag": 0.0, "torque_mag": 0.0, "timestamp": 0.0},
        )

        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=list(dummy.keys())
        )
        self._csv_writer.writeheader()

        self._meta = {
            "episode_id":  episode_id,
            "object_name": object_name,
            "scenario":    scenario,
            "started_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "csv_path":    str(csv_path),
            **(extra_meta or {}),
        }

    def log(
        self,
        sim_time:   float,
        axes_dicts: list[dict],   # from AxisMonitor.snapshot()
        contact:    dict,         # ContactState.to_dict()
        tactile:    dict,         # TactileReading.to_dict() — without taxel_pressures
        ft:         dict,         # FTReading.to_dict()
    ) -> None:
        if self._csv_writer is None:
            raise RuntimeError("Call start() before log().")

        # Strip large taxel array — logged separately if needed
        tactile_summary = {k: v for k, v in tactile.items() if k != "taxel_pressures"}

        row = self._build_row(sim_time, axes_dicts, contact, tactile_summary, ft)
        self._csv_writer.writerow(row)
        self._row_count += 1

    def stop(
        self,
        outcome: str = "unknown",
        metrics: dict | None = None,
    ) -> Path:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

        self._meta.update({
            "outcome":    outcome,
            "row_count":  self._row_count,
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics":    metrics or {},
        })

        meta_path = self.log_dir / f"{self._episode_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

        print(f"[DataLogger] Saved {self._row_count} rows → {self._meta['csv_path']}")
        return meta_path

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_row(
        sim_time:       float,
        axes_dicts:     list[dict],
        contact_dict:   dict,
        tactile_dict:   dict,
        ft_dict:        dict,
    ) -> dict:
        row: dict = {"t": round(sim_time, 4)}

        for i, ax in enumerate(axes_dicts):
            j = i + 1
            row[f"j{j}_pos"]    = ax.get("position", 0.0)
            row[f"j{j}_vel"]    = ax.get("velocity", 0.0)
            row[f"j{j}_active"] = ax.get("is_active", 0)

        row.update({f"contact_{k}": v for k, v in contact_dict.items()})
        row.update({f"tactile_{k}": v for k, v in tactile_dict.items()})
        row.update({f"ft_{k}":      v for k, v in ft_dict.items()})

        return row