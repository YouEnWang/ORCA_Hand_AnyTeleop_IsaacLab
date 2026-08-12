#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline self-consistency test for ORCA vector retargeting.

Procedure:
1. Set ORCA q = 0.
2. Use ORCA forward kinematics to compute its own task vectors.
3. Convert those robot vectors back into a synthetic human reference.
4. Feed the reference into VectorOptimizer.
5. Check whether the output remains close to q = 0.

No camera and no Isaac Sim are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    config_path = args.config.resolve()

    cfg = RetargetingConfig.load_from_file(config_path)
    retargeting = cfg.build()

    optimizer = retargeting.optimizer
    robot = optimizer.robot

    print("=" * 80)
    print("ORCA offline vector-retargeting self test")
    print("=" * 80)

    # ------------------------------------------------------------
    # Reference robot state
    # ------------------------------------------------------------

    q_reference = np.zeros(
        robot.dof,
        dtype=np.float32,
    )

    # Set optimizer initial state to the same pose.
    # This avoids regularization pulling the solution toward an unrelated
    # initial joint configuration.
    retargeting.set_qpos(q_reference)

    # ------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------

    robot.compute_forward_kinematics(q_reference)

    link_poses = [
        robot.get_link_pose(index)
        for index in optimizer.computed_link_indices
    ]

    body_positions = np.asarray(
        [pose[:3, 3] for pose in link_poses],
        dtype=np.float32,
    )

    origin_indices = (
        optimizer.origin_link_indices
        .cpu()
        .numpy()
    )

    task_indices = (
        optimizer.task_link_indices
        .cpu()
        .numpy()
    )

    origin_positions = body_positions[
        origin_indices
    ]

    task_positions = body_positions[
        task_indices
    ]

    robot_vectors = (
        task_positions
        -
        origin_positions
    )

    # VectorOptimizer internally multiplies human reference vectors
    # by scaling_factor. Therefore:
    #
    #   human_ref * scaling = robot_vector
    #
    reference_vectors = (
        robot_vectors
        /
        float(cfg.scaling_factor)
    )

    # ------------------------------------------------------------
    # Fixed joints
    # ------------------------------------------------------------

    fixed_qpos = q_reference[
        optimizer.idx_pin2fixed
    ]

    print()
    print("Robot DOF        :", robot.dof)
    print("Optimized DOF    :", optimizer.opt_dof)
    print("Fixed joints     :", optimizer.fixed_joint_names)
    print("Reference shape  :", reference_vectors.shape)
    print("Fixed qpos       :", fixed_qpos)

    # ------------------------------------------------------------
    # Retarget
    # ------------------------------------------------------------

    q_output = retargeting.retarget(
        reference_vectors,
        fixed_qpos=fixed_qpos,
    )

    print()
    print("Output qpos:")
    print(q_output)

    print()
    print("Finite:", np.all(np.isfinite(q_output)))

    max_error = float(
        np.max(
            np.abs(
                q_output - q_reference
            )
        )
    )

    print(
        "Maximum difference from reference:",
        max_error,
        "rad",
    )

    if not np.all(np.isfinite(q_output)):
        raise RuntimeError(
            "Retargeting produced NaN or Inf."
        )

    if max_error < 0.1:
        print(
            "[PASS] Offline retargeting "
            "self-consistency test succeeded."
        )
    else:
        print(
            "[WARNING] Retargeting succeeded, "
            "but result differs from reference by "
            f"{max_error:.6f} rad."
        )

    print()
    retargeting.verbose()


if __name__ == "__main__":
    main()
