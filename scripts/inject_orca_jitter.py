#!/usr/bin/env python3
"""Create a controlled noisy ORCA demonstration from a clean trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.io import load_dataset, normalize_time, require_trajectory, save_npz
from research.orca_research.noise import inject_jitter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-key", default="q_command")
    parser.add_argument("--amplitude-rad", type=float, default=0.015)
    parser.add_argument("--frequency-hz", type=float, default=12.0)
    parser.add_argument("--gaussian-std-rad", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    time_values = normalize_time(np.asarray(dataset["time"], dtype=np.float64))
    clean = require_trajectory(dataset, args.trajectory_key)
    noisy, noise = inject_jitter(
        clean,
        time_values,
        amplitude_rad=args.amplitude_rad,
        frequency_hz=args.frequency_hz,
        gaussian_std_rad=args.gaussian_std_rad,
        seed=args.seed,
    )
    dataset[f"{args.trajectory_key}_clean"] = clean
    dataset[f"{args.trajectory_key}_noise"] = noise
    dataset[args.trajectory_key] = noisy
    save_npz(args.output, dataset)
    print(f"Wrote noisy dataset to {args.output}")


if __name__ == "__main__":
    main()
