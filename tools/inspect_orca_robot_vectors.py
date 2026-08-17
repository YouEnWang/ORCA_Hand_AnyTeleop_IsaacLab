#!/usr/bin/env python3

"""
inspect_orca_robot_vectors.py

Purpose
-------
Inspect the robot-side reference vectors used by the
dex-retargeting VectorOptimizer for ORCA Hand v2 Right.

This script:

1. Loads the same ORCA URDF used by dex-retargeting.
2. Loads the custom ORCA vector-retargeting YAML.
3. Sets all robot joints to q = 0.
4. Computes forward kinematics with Pinocchio.
5. Finds the origin/task link frames used by VectorOptimizer.
6. Computes:

       v_robot_i
           = p_task_i - p_origin_i

7. Prints all 10 robot reference vectors:

       vx, vy, vz, length

IMPORTANT
---------
This is a diagnostic script only.

It does NOT:
- modify the URDF,
- modify the retargeting configuration,
- run optimization,
- send UDP,
- control Isaac Sim.

The purpose is to compare the ORCA robot geometry against
the human reference vectors measured in Step 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import yaml


# ==============================================================
# Utility
# ==============================================================


def find_key_recursive(
    obj: Any,
    target_key: str,
):
    """
    Search recursively through a nested YAML structure.

    This makes the script tolerant to whether the retargeting
    fields are placed at the YAML root or inside another section.
    """

    if isinstance(obj, dict):

        if target_key in obj:
            return obj[target_key]

        for value in obj.values():

            result = find_key_recursive(
                value,
                target_key,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:

            result = find_key_recursive(
                value,
                target_key,
            )

            if result is not None:
                return result

    return None


def find_frame_id_by_name(
    model: pin.Model,
    frame_name: str,
) -> int:
    """
    Find an exact Pinocchio frame by name.

    We scan model.frames explicitly instead of depending on
    version-specific getFrameId behavior.
    """

    matches = []

    for frame_id, frame in enumerate(
        model.frames
    ):

        if frame.name == frame_name:

            matches.append(
                frame_id
            )

    if len(matches) == 0:

        raise RuntimeError(
            "\n"
            f"Cannot find Pinocchio frame:\n"
            f"    {frame_name}\n"
        )

    if len(matches) > 1:

        print(
            "[WARNING] Multiple Pinocchio frames "
            f"named '{frame_name}'. "
            f"Using frame ID {matches[0]}."
        )

    return matches[0]


def get_frame_position(
    model: pin.Model,
    data: pin.Data,
    frame_name: str,
) -> tuple[int, np.ndarray]:
    """
    Return the world-frame translation of a named frame.
    """

    frame_id = find_frame_id_by_name(
        model,
        frame_name,
    )

    position = np.asarray(
        data.oMf[
            frame_id
        ].translation,
        dtype=np.float64,
    ).copy()

    return (
        frame_id,
        position,
    )


# ==============================================================
# Main
# ==============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help=(
            "Path to the ORCA URDF used by "
            "dex-retargeting."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to ORCA VECTOR retargeting YAML."
        ),
    )

    args = parser.parse_args()


    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    urdf_path = Path(
        args.urdf
    ).expanduser().resolve()

    config_path = Path(
        args.config
    ).expanduser().resolve()


    if not urdf_path.exists():

        raise FileNotFoundError(
            f"URDF does not exist: "
            f"{urdf_path}"
        )


    if not config_path.exists():

        raise FileNotFoundError(
            f"Config does not exist: "
            f"{config_path}"
        )


    # ----------------------------------------------------------
    # Load YAML
    # ----------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )


    origin_link_names = (
        find_key_recursive(
            config,
            "target_origin_link_names",
        )
    )

    task_link_names = (
        find_key_recursive(
            config,
            "target_task_link_names",
        )
    )


    if origin_link_names is None:

        raise KeyError(
            "Cannot find "
            "'target_origin_link_names' "
            "in YAML."
        )


    if task_link_names is None:

        raise KeyError(
            "Cannot find "
            "'target_task_link_names' "
            "in YAML."
        )


    origin_link_names = list(
        origin_link_names
    )

    task_link_names = list(
        task_link_names
    )


    if (
        len(origin_link_names)
        != len(task_link_names)
    ):

        raise RuntimeError(
            "Origin/task link count mismatch:\n"
            f"origin = {len(origin_link_names)}\n"
            f"task   = {len(task_link_names)}"
        )


    number_of_vectors = len(
        origin_link_names
    )


    # ----------------------------------------------------------
    # Load robot model
    # ----------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "ORCA Robot Reference Vector Diagnostics"
    )

    print(
        "=" * 80
    )

    print(
        f"URDF   : {urdf_path}"
    )

    print(
        f"Config : {config_path}"
    )

    print()


    model = pin.buildModelFromUrdf(
        str(
            urdf_path
        )
    )

    data = model.createData()


    print(
        f"Robot name       : {model.name}"
    )

    print(
        f"Pinocchio nq     : {model.nq}"
    )

    print(
        f"Pinocchio nv     : {model.nv}"
    )

    print(
        f"Pinocchio joints : "
        f"{model.njoints}"
    )

    print(
        f"Pinocchio frames : "
        f"{len(model.frames)}"
    )

    print(
        f"Vector count     : "
        f"{number_of_vectors}"
    )

    print()


    # ----------------------------------------------------------
    # Explicit q = 0
    # ----------------------------------------------------------

    q_zero = np.zeros(
        model.nq,
        dtype=np.float64,
    )


    print(
        "=" * 80
    )

    print(
        "Robot configuration"
    )

    print(
        "=" * 80
    )

    print(
        "Using explicit q = 0:"
    )

    print(
        q_zero
    )

    print()


    # ----------------------------------------------------------
    # FK
    # ----------------------------------------------------------

    pin.forwardKinematics(
        model,
        data,
        q_zero,
    )

    pin.updateFramePlacements(
        model,
        data,
    )


    # ----------------------------------------------------------
    # Print mapping first
    # ----------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "Robot reference-vector mapping"
    )

    print(
        "=" * 80
    )


    for vector_index, (
        origin_name,
        task_name,
    ) in enumerate(
        zip(
            origin_link_names,
            task_link_names,
        )
    ):

        print(
            f"[RV{vector_index:02d}] "
            f"{origin_name} "
            f"-> "
            f"{task_name}"
        )


    print()


    # ----------------------------------------------------------
    # Compute vectors
    # ----------------------------------------------------------

    robot_vectors = []

    robot_lengths = []


    print(
        "=" * 80
    )

    print(
        "ORCA robot vectors at q = 0"
    )

    print(
        "=" * 80
    )


    for vector_index, (
        origin_name,
        task_name,
    ) in enumerate(
        zip(
            origin_link_names,
            task_link_names,
        )
    ):

        (
            origin_frame_id,
            origin_position,
        ) = get_frame_position(
            model,
            data,
            origin_name,
        )

        (
            task_frame_id,
            task_position,
        ) = get_frame_position(
            model,
            data,
            task_name,
        )


        robot_vector = (
            task_position
            -
            origin_position
        )


        length = float(
            np.linalg.norm(
                robot_vector
            )
        )


        robot_vectors.append(
            robot_vector
        )

        robot_lengths.append(
            length
        )


        vx = float(
            robot_vector[0]
        )

        vy = float(
            robot_vector[1]
        )

        vz = float(
            robot_vector[2]
        )


        print(
            f"[RV{vector_index:02d}] "
            f"{origin_name} "
            f"-> "
            f"{task_name}"
        )

        print(
            f"       origin_frame_id="
            f"{origin_frame_id:3d}  "
            f"task_frame_id="
            f"{task_frame_id:3d}"
        )

        print(
            f"       origin="
            f"[{origin_position[0]:+.6f}, "
            f"{origin_position[1]:+.6f}, "
            f"{origin_position[2]:+.6f}]"
        )

        print(
            f"       task  ="
            f"[{task_position[0]:+.6f}, "
            f"{task_position[1]:+.6f}, "
            f"{task_position[2]:+.6f}]"
        )

        print(
            f"       v="
            f"[{vx:+.6f}, "
            f"{vy:+.6f}, "
            f"{vz:+.6f}]  "
            f"L={length:.6f}"
        )

        print()


    robot_vectors = np.asarray(
        robot_vectors,
        dtype=np.float64,
    )

    robot_lengths = np.asarray(
        robot_lengths,
        dtype=np.float64,
    )


    # ----------------------------------------------------------
    # Compact matrix
    # ----------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "Robot vector matrix B (10 x 3)"
    )

    print(
        "=" * 80
    )


    np.set_printoptions(
        precision=6,
        suppress=True,
    )


    print(
        robot_vectors
    )

    print()


    print(
        "=" * 80
    )

    print(
        "Robot vector lengths"
    )

    print(
        "=" * 80
    )


    for vector_index, length in enumerate(
        robot_lengths
    ):

        print(
            f"RV{vector_index:02d}: "
            f"{length:.6f}"
        )


    print()


    # ----------------------------------------------------------
    # Basic geometry diagnostics
    # ----------------------------------------------------------

    mean_length = float(
        robot_lengths.mean()
    )

    min_length = float(
        robot_lengths.min()
    )

    max_length = float(
        robot_lengths.max()
    )


    print(
        "=" * 80
    )

    print(
        "Summary"
    )

    print(
        "=" * 80
    )

    print(
        f"Mean vector length : "
        f"{mean_length:.6f}"
    )

    print(
        f"Min vector length  : "
        f"{min_length:.6f}"
    )

    print(
        f"Max vector length  : "
        f"{max_length:.6f}"
    )

    print()

    print(
        "DONE."
    )


if __name__ == "__main__":
    main()
