#!/usr/bin/env python3
"""Replay a saved ORCA trajectory through UDP to the IsaacLab client."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.orca_research.io import load_dataset, normalize_time, require_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trajectory-key", default="q_command")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    time_values = normalize_time(np.asarray(dataset["time"], dtype=np.float64))
    q = require_trajectory(dataset, args.trajectory_key)
    joint_names = list(dataset.get("joint_names", []))
    if not joint_names:
        joint_names = [f"joint_{idx:02d}" for idx in range(q.shape[1])]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    try:
        while True:
            start = time.monotonic()
            for frame in range(q.shape[0]):
                if frame > 0:
                    dt = (time_values[frame] - time_values[frame - 1]) / max(args.rate_scale, 1e-9)
                    target_time = start + time_values[frame] / max(args.rate_scale, 1e-9)
                    sleep_time = max(target_time - time.monotonic(), min(dt, 0.0))
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                packet = {
                    "seq": seq,
                    "timestamp": time.time(),
                    "tracking_valid": True,
                    "joint_names": joint_names,
                    "positions_rad": q[frame].astype(float).tolist(),
                    "source": "orca_demo_replay",
                }
                sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))
                seq += 1
            if not args.loop:
                break
    finally:
        sock.close()


if __name__ == "__main__":
    main()
