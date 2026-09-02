#!/usr/bin/env python3
"""Generate a tiny Isaac-client-style JSONL file for loader smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_synthetic_orca_demo import DEFAULT_JOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/examples/sample_isaac_client.jsonl"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--rate", type=float, default=120.0)
    args = parser.parse_args()

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as file:
        q_actual = np.zeros(len(DEFAULT_JOINT_NAMES), dtype=np.float64)
        for frame in range(args.frames):
            t_sim = frame / args.rate
            q_desired = 0.2 * np.sin(2.0 * np.pi * 0.5 * t_sim + np.arange(len(DEFAULT_JOINT_NAMES)) * 0.05)
            q_command = q_desired.copy()
            q_actual = q_actual + 0.2 * (q_command - q_actual)
            record = {
                "t_sim": t_sim,
                "timestamp": 1_800_000_000.0 + t_sim,
                "seq": frame,
                "packet_timestamp": 1_800_000_000.0 + t_sim,
                "tracking_valid": True,
                "joint_names": DEFAULT_JOINT_NAMES,
                "q_desired": q_desired.astype(float).tolist(),
                "q_command": q_command.astype(float).tolist(),
                "q_actual": q_actual.astype(float).tolist(),
                "source": "sample_isaaclab_orca_anyteleop_client",
            }
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            file.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

