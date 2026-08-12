#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send a simple ORCA joint command from the retargeting container
to the Isaac Lab AnyTeleop client.

This verifies:
- dex-retargeting robot model
- 17 joint names
- UDP communication
- joint-name mapping
- Isaac position control

No MediaPipe is used.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import (
    RetargetingConfig,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5006,
    )

    parser.add_argument(
        "--joint",
        type=str,
        default=(
            "I-PP_bacbd481_to_"
            "I-AP-R_d95d02d1"
        ),
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    cfg = RetargetingConfig.load_from_file(
        args.config.resolve()
    )

    retargeting = cfg.build()

    robot = retargeting.optimizer.robot

    joint_names = list(
        retargeting.joint_names
    )

    if len(joint_names) != 17:
        raise RuntimeError(
            f"Expected 17 joints, "
            f"got {len(joint_names)}"
        )

    if args.joint not in joint_names:
        raise ValueError(
            f"Joint not found: {args.joint}"
        )

    joint_index = joint_names.index(
        args.joint
    )

    qpos = np.zeros(
        robot.dof,
        dtype=np.float32,
    )

    joint_limits = np.asarray(
        robot.joint_limits
    )

    requested = (
        qpos[joint_index]
        + args.offset
    )

    qpos[joint_index] = np.clip(
        requested,
        joint_limits[joint_index, 0],
        joint_limits[joint_index, 1],
    )

    print("=" * 80)
    print("ORCA static UDP test")
    print("=" * 80)

    print("Target joint :", args.joint)
    print("Index        :", joint_index)
    print(
        "Command      :",
        qpos[joint_index],
        "rad",
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    period = 1.0 / args.rate

    start = time.monotonic()

    seq = 0

    while (
        time.monotonic() - start
        < args.duration
    ):

        packet = {
            "seq": seq,
            "timestamp": time.time(),
            "tracking_valid": True,
            "joint_names": joint_names,
            "positions_rad": (
                qpos.astype(float).tolist()
            ),
            "source": "static_test",
        }

        encoded = json.dumps(
            packet
        ).encode("utf-8")

        sock.sendto(
            encoded,
            (
                args.host,
                args.port,
            ),
        )

        seq += 1

        time.sleep(period)

    sock.close()

    print(
        "[PASS] Static command "
        "transmission completed."
    )


if __name__ == "__main__":
    main()
