#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test whether ORCA Hand v2 Right can be loaded by dex-retargeting.

This script does NOT:
- use a camera
- run MediaPipe
- connect to Isaac Sim
- command any robot

It only verifies:
1. ORCA URDF parsing
2. Pinocchio model construction
3. ORCA joint names
4. Optimized/fixed joints
5. Vector retargeting configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig


EXPECTED_DOF = 17

EXPECTED_WRIST = (
    "R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e"
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config_path = args.config.resolve()

    if not config_path.is_file():
        print(
            f"[ERROR] Config does not exist: {config_path}",
            file=sys.stderr,
        )
        return 2

    print("=" * 80)
    print("ORCA dex-retargeting configuration test")
    print("=" * 80)

    print(f"Config: {config_path}")

    # ------------------------------------------------------------
    # 1. Load YAML
    # ------------------------------------------------------------

    cfg = RetargetingConfig.load_from_file(config_path)

    print()
    print("[1] Retargeting configuration")
    print(f"Type              : {cfg.type}")
    print(f"URDF              : {cfg.urdf_path}")
    print(f"Scaling factor    : {cfg.scaling_factor}")
    print(f"Joint limits      : {cfg.has_joint_limits}")
    print(f"Low-pass alpha    : {cfg.low_pass_alpha}")

    # ------------------------------------------------------------
    # 2. Build Pinocchio + optimizer
    # ------------------------------------------------------------

    print()
    print("[2] Building dex-retargeting...")

    retargeting = cfg.build()

    print("[PASS] Retargeting build succeeded.")

    optimizer = retargeting.optimizer
    robot = optimizer.robot

    # ------------------------------------------------------------
    # 3. Robot model inspection
    # ------------------------------------------------------------

    print()
    print("[3] Robot model")

    print(f"Robot DOF         : {robot.dof}")
    print(f"Optimized DOF     : {optimizer.opt_dof}")
    print(f"Fixed DOF         : {len(optimizer.idx_pin2fixed)}")

    print()
    print("Full robot joint order:")

    for index, name in enumerate(retargeting.joint_names):
        print(f"  [{index:02d}] {name}")

    # ------------------------------------------------------------
    # 4. Optimized/fixed joints
    # ------------------------------------------------------------

    print()
    print("[4] Optimized joints")

    for index, name in enumerate(optimizer.target_joint_names):
        print(f"  [{index:02d}] {name}")

    print()
    print("[5] Fixed joints")

    fixed_joint_names = optimizer.fixed_joint_names

    for index, name in enumerate(fixed_joint_names):
        print(f"  [{index:02d}] {name}")

    # ------------------------------------------------------------
    # 5. Task vectors
    # ------------------------------------------------------------

    print()
    print("[6] Human / robot vector configuration")

    indices = optimizer.target_link_human_indices

    print("Human index shape :", indices.shape)
    print("Human indices:")
    print(indices)

    print()

    for i, (origin, task) in enumerate(
        zip(
            optimizer.origin_link_names,
            optimizer.task_link_names,
        )
    ):
        human_origin = indices[0, i]
        human_task = indices[1, i]

        print(
            f"  Vector {i:02d}: "
            f"human[{human_origin}] -> human[{human_task}]"
        )

        print(
            f"             {origin} -> {task}"
        )

    # ------------------------------------------------------------
    # 6. Assertions
    # ------------------------------------------------------------

    print()
    print("[7] Sanity checks")

    errors = []

    if robot.dof != EXPECTED_DOF:
        errors.append(
            f"Expected {EXPECTED_DOF} DoF, got {robot.dof}"
        )

    if len(optimizer.target_joint_names) != 16:
        errors.append(
            "Expected 16 optimized finger joints."
        )

    if len(fixed_joint_names) != 1:
        errors.append(
            f"Expected exactly 1 fixed joint, "
            f"got {len(fixed_joint_names)}"
        )

    elif fixed_joint_names[0] != EXPECTED_WRIST:
        errors.append(
            "Fixed joint is not the expected ORCA wrist:\n"
            f"  expected: {EXPECTED_WRIST}\n"
            f"  actual  : {fixed_joint_names[0]}"
        )

    if indices.shape != (2, 10):
        errors.append(
            f"Expected human index shape (2, 10), "
            f"got {indices.shape}"
        )

    if errors:
        print()
        for error in errors:
            print(f"[FAIL] {error}")

        return 1

    print("[PASS] ORCA has 17 DoF.")
    print("[PASS] 16 finger joints are optimized.")
    print("[PASS] Only ORCA wrist is fixed.")
    print("[PASS] 10 vector constraints are configured.")

    print()
    print("=" * 80)
    print("ORCA dex-retargeting configuration is valid.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
