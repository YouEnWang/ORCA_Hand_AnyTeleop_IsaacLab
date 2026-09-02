"""Small reporting helpers for metrics tables and figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import ensure_parent


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    serializable = _to_serializable(data)
    path.expanduser().write_text(
        json.dumps(serializable, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        path.expanduser().write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.expanduser().open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectory_overlay(
    path: Path,
    time_values: np.ndarray,
    raw: np.ndarray,
    purified: np.ndarray | None,
    joint_names: list[str],
    max_joints: int = 6,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ensure_parent(path)
    joint_count = min(raw.shape[1], max_joints)
    fig, axes = plt.subplots(joint_count, 1, figsize=(10, 1.8 * joint_count), sharex=True)
    if joint_count == 1:
        axes = [axes]
    for idx, axis in enumerate(axes):
        name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx:02d}"
        axis.plot(time_values, raw[:, idx], label="raw", linewidth=1.2)
        if purified is not None:
            axis.plot(time_values, purified[:, idx], label="purified", linewidth=1.2)
        axis.set_ylabel(name, fontsize=8)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path.expanduser(), dpi=160)
    plt.close(fig)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value

