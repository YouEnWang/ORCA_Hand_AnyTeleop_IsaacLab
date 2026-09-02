#!/usr/bin/env python3
"""Generate a small synthetic ORCA-like joint trajectory for smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.io import save_npz


DEFAULT_JOINT_NAMES = [
    "wrist",
    "index_abd",
    "index_mcp",
    "index_pip",
    "middle_abd",
    "middle_mcp",
    "middle_pip",
    "ring_abd",
    "ring_mcp",
    "ring_pip",
    "pinky_abd",
    "pinky_mcp",
    "pinky_pip",
    "thumb_cmc",
    "thumb_abd",
    "thumb_mcp",
    "thumb_dip",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/examples/synthetic_orca_demo.npz"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--rate", type=float, default=120.0)
    args = parser.parse_args()

    frame_count = int(args.duration * args.rate)
    time_values = np.arange(frame_count, dtype=np.float64) / args.rate
    joint_count = len(DEFAULT_JOINT_NAMES)

    q_desired = np.zeros((frame_count, joint_count), dtype=np.float64)
    for joint_index in range(joint_count):
        frequency = 0.15 + 0.02 * joint_index
        amplitude = 0.25 if joint_index > 0 else 0.05
        q_desired[:, joint_index] = amplitude * (
            0.5 - 0.5 * np.cos(2.0 * np.pi * frequency * time_values)
        )

    q_command = q_desired.copy()
    q_actual = np.zeros_like(q_command)
    alpha = 0.18
    for frame in range(1, frame_count):
        q_actual[frame] = q_actual[frame - 1] + alpha * (q_command[frame] - q_actual[frame - 1])

    save_npz(
        args.output,
        {
            "time": time_values,
            "joint_names": DEFAULT_JOINT_NAMES,
            "q_desired": q_desired,
            "q_command": q_command,
            "q_actual": q_actual,
            "tracking_valid": np.ones(frame_count, dtype=bool),
        },
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
