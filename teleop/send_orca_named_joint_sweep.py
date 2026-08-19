#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import socket
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def load_semantics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {
        semantic: info["urdf"]
        for semantic, info
        in data["joints"].items()
    }


def load_urdf_joints(path):

    root = ET.parse(path).getroot()

    joint_names = []
    limits = {}

    for joint in root.findall("joint"):

        if joint.attrib.get("type") != "revolute":
            continue

        name = joint.attrib["name"]

        limit = joint.find("limit")

        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])

        joint_names.append(name)
        limits[name] = (lower, upper)

    return joint_names, limits


def send_pose(
    sock,
    host,
    port,
    joint_names,
    q,
    seq,
):

    packet = {
        "seq": seq,
        "timestamp": time.time(),
        "tracking_valid": True,
        "joint_names": joint_names,
        "positions_rad": q.tolist(),
        "source": "orca_named_joint_sweep",
    }

    sock.sendto(
        json.dumps(packet).encode("utf-8"),
        (host, port),
    )


def hold_pose(
    sock,
    host,
    port,
    joint_names,
    q,
    seq,
    duration,
    rate,
):

    dt = 1.0 / rate
    end_time = time.monotonic() + duration

    while time.monotonic() < end_time:

        send_pose(
            sock,
            host,
            port,
            joint_names,
            q,
            seq,
        )

        seq += 1

        time.sleep(dt)

    return seq


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urdf",
        required=True,
    )

    parser.add_argument(
        "--semantics",
        required=True,
    )

    parser.add_argument(
        "--joint",
        required=True,
        help="Semantic joint name, e.g. index_abd",
    )

    parser.add_argument(
        "--amplitude-deg",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5006,
    )

    args = parser.parse_args()


    semantics = load_semantics(
        args.semantics
    )

    if args.joint not in semantics:
        raise KeyError(
            f"Unknown semantic joint: {args.joint}"
        )


    joint_names, limits = load_urdf_joints(
        args.urdf
    )


    selected_name = semantics[
        args.joint
    ]

    if selected_name not in joint_names:
        raise RuntimeError(
            f"Joint not found in URDF: {selected_name}"
        )


    selected_index = joint_names.index(
        selected_name
    )

    lower, upper = limits[
        selected_name
    ]


    amplitude = np.deg2rad(
        args.amplitude_deg
    )


    q_zero = np.zeros(
        len(joint_names),
        dtype=np.float64,
    )


    positive = float(
        np.clip(
            amplitude,
            lower,
            upper,
        )
    )

    negative = float(
        np.clip(
            -amplitude,
            lower,
            upper,
        )
    )


    print("=" * 80)
    print("ORCA named joint sweep")
    print("=" * 80)

    print("Semantic :", args.joint)
    print("URDF     :", selected_name)
    print("Index    :", selected_index)
    print(
        "Range    :",
        f"[{lower:+.4f}, {upper:+.4f}] rad",
    )

    print(
        "Test     :",
        f"0 -> {positive:+.4f} "
        f"-> 0 -> {negative:+.4f} -> 0",
    )


    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    # seq = 0
    # Use wall-clock milliseconds as the initial sequence number.
    #
    # The Isaac client keeps last_seq across UDP sender restarts.
    # Starting every new sweep from seq=0 would therefore cause
    # packets from the second sweep onward to be rejected as stale.
    seq = int(time.time() * 1000)


    sequence = [
        ("ZERO", 0.0),
        ("POSITIVE", positive),
        ("ZERO", 0.0),
        ("NEGATIVE", negative),
        ("ZERO", 0.0),
    ]


    for label, value in sequence:

        q = q_zero.copy()

        q[selected_index] = value

        print(
            f"[{label}] "
            f"{args.joint}={value:+.4f}"
        )

        seq = hold_pose(
            sock,
            args.host,
            args.port,
            joint_names,
            q,
            seq,
            args.hold,
            args.rate,
        )


    sock.close()

    print("DONE.")


if __name__ == "__main__":
    main()
