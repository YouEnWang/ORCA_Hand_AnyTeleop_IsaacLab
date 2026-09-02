#!/usr/bin/env python3
"""Compare raw and purified ORCA demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.io import load_dataset, normalize_time, require_trajectory
from research.orca_research.metrics import compute_quality_metrics, summarize_joint_metrics
from research.orca_research.reporting import plot_trajectory_overlay, write_csv, write_json


def _prefix_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--purified", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw_vs_purified"))
    parser.add_argument("--trajectory-key", default="q_command")
    args = parser.parse_args()

    raw_dataset = load_dataset(args.raw)
    purified_dataset = load_dataset(args.purified)
    time_values = normalize_time(np.asarray(raw_dataset["time"], dtype=np.float64))
    raw = require_trajectory(raw_dataset, args.trajectory_key)
    purified = require_trajectory(purified_dataset, args.trajectory_key)
    if raw.shape != purified.shape:
        raise ValueError(f"Trajectory shape mismatch: {raw.shape} vs {purified.shape}")

    raw_metrics = compute_quality_metrics(time_values, raw)
    purified_metrics = compute_quality_metrics(time_values, purified)
    delta = {
        "rmse_raw_minus_purified": np.sqrt(np.mean((raw - purified) ** 2, axis=0)),
    }
    all_metrics = {}
    all_metrics.update(_prefix_metrics("raw", raw_metrics))
    all_metrics.update(_prefix_metrics("purified", purified_metrics))
    all_metrics.update(delta)

    joint_names = list(raw_dataset.get("joint_names", []))
    rows = summarize_joint_metrics(joint_names, all_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_vs_purified_joint_metrics.csv", rows)
    write_json(args.output_dir / "raw_vs_purified_summary.json", all_metrics)
    plot_trajectory_overlay(
        args.output_dir / "raw_vs_purified_preview.png",
        time_values,
        raw,
        purified,
        joint_names,
    )
    print(f"Wrote comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
