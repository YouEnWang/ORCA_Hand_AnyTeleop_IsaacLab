#!/usr/bin/env python3
"""Analyze ORCA teleoperation demonstration quality metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.io import load_dataset, normalize_time, require_trajectory
from research.orca_research.metrics import (
    QualityConfig,
    compute_quality_metrics,
    summarize_joint_metrics,
    tracking_error_metrics,
)
from research.orca_research.reporting import plot_trajectory_overlay, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/quality"))
    parser.add_argument("--trajectory-key", default="q_command")
    parser.add_argument("--high-frequency-cutoff-hz", type=float, default=6.0)
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    time_values = normalize_time(np.asarray(dataset["time"], dtype=np.float64))
    q = require_trajectory(dataset, args.trajectory_key)
    joint_names = list(dataset.get("joint_names", []))

    metrics = compute_quality_metrics(
        time_values,
        q,
        QualityConfig(high_frequency_cutoff_hz=args.high_frequency_cutoff_hz),
    )

    if "q_actual" in dataset and args.trajectory_key in ("q_command", "q_desired"):
        reference = require_trajectory(dataset, args.trajectory_key)
        actual = require_trajectory(dataset, "q_actual")
        metrics.update(tracking_error_metrics(reference, actual))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_joint_metrics(joint_names, metrics)
    write_csv(args.output_dir / "joint_quality_metrics.csv", rows)
    write_json(args.output_dir / "quality_summary.json", metrics)
    plot_trajectory_overlay(
        args.output_dir / "trajectory_preview.png",
        time_values,
        q,
        None,
        joint_names,
    )
    print(f"Wrote metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
