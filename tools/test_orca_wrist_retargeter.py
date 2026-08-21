#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic smoke test for orca_wrist_retargeter.py."""

import math
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
TELEOP_DIR = THIS.parents[1] / "teleop"
sys.path.insert(0, str(TELEOP_DIR))

from orca_wrist_retargeter import twist_angle_about_axis


def rot_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rot_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rot_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


axis_x = np.asarray([1.0, 0.0, 0.0])

cases = [
    ("Rx(+20 deg)", rot_x(math.radians(+20.0)), +20.0),
    ("Rx(-20 deg)", rot_x(math.radians(-20.0)), -20.0),
    ("Ry(+20 deg)", rot_y(math.radians(+20.0)), 0.0),
    ("Rz(+20 deg)", rot_z(math.radians(+20.0)), 0.0),
]

failed = False

for name, rotation, expected_deg in cases:
    actual_rad = twist_angle_about_axis(
        rotation,
        axis_x,
    )

    actual_deg = math.degrees(
        actual_rad
    )

    error_deg = abs(
        actual_deg
        -
        expected_deg
    )

    print(
        f"{name:<14} "
        f"actual={actual_deg:+9.4f} deg  "
        f"expected={expected_deg:+9.4f} deg  "
        f"error={error_deg:.6f}"
    )

    if error_deg > 1e-4:
        failed = True

if failed:
    raise SystemExit(
        "FAIL: wrist twist math."
    )

print("PASS: wrist twist math.")
