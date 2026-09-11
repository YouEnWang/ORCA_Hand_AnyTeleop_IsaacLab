#!/usr/bin/env python3
"""Purify an ORCA demonstration trajectory with a selected baseline method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.filters import apply_filter
from research.orca_research.io import load_dataset, normalize_time, require_trajectory, save_npz
from research.orca_research.reporting import plot_trajectory_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-key", default="q_command")
    parser.add_argument(
        "--method",
        default="butterworth",
        choices=[
            "none",
            "moving_average",
            "median",
            "savgol",
            "butterworth",
            "outlier_then_butterworth",
            "velocity_clamp",
            "tracking_interp",
            "tracking_interp_then_butterworth",
        ],
    )
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--cutoff-hz", type=float, default=6.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--z-threshold", type=float, default=6.0)
    parser.add_argument("--max-speed-rad-s", type=float, default=2.0)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    time_values = normalize_time(np.asarray(dataset["time"], dtype=np.float64))
    raw = require_trajectory(dataset, args.trajectory_key)
    tracking_valid = dataset.get("tracking_valid",None)     # Tracking Dropout
    purified = apply_filter(
        raw,
        time_values,
        method=args.method,
        window=args.window,
        polyorder=args.polyorder,
        cutoff_hz=args.cutoff_hz,
        order=args.order,
        z_threshold=args.z_threshold,
        max_speed_rad_s=args.max_speed_rad_s,
        tracking_valid=tracking_valid,
    )
    dataset[f"{args.trajectory_key}_raw"] = raw
    dataset[args.trajectory_key] = purified
    dataset["purification_method"] = np.asarray(args.method)
    dataset["purification_used_tracking_valid"] = np.asarray(args.method in {"tracking_interp", "tracking_interp_then_butterworth"})     # Tracking Dropout

    save_npz(args.output, dataset)

    if args.figure is not None:
        plot_trajectory_overlay(
            args.figure,
            time_values,
            raw,
            purified,
            list(dataset.get("joint_names", [])),
        )
    print(f"Wrote purified dataset to {args.output}")


if __name__ == "__main__":
    main()
