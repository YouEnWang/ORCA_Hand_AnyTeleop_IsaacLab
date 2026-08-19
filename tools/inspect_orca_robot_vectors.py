#!/usr/bin/env python3

"""
inspect_orca_robot_vectors.py

Purpose
-------
Inspect the robot-side reference vectors used by the
dex-retargeting VectorOptimizer for ORCA Hand v2 Right.

This script:

1. Loads an ORCA URDF with Pinocchio.
2. Loads the VECTOR retargeting YAML.
3. Sets all movable robot joints to q = 0.
4. Computes forward kinematics.
5. Reads:

       target_origin_link_names
       target_task_link_names

   from the retargeting YAML.

6. Computes every robot reference vector:

       v_robot_i
           = p_task_i - p_origin_i

7. Prints:
       - origin frame
       - task frame
       - origin position
       - task position
       - vector xyz
       - vector length

This works with both:

    orca_v2_right_vector.yml

and the Step-7 virtual-tip version:

    orca_v2_right_vector_virtual_tip.yml

Diagnostic only:
- does not run optimization
- does not control Isaac
- does not send UDP
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import yaml


# ==============================================================
# YAML utilities
# ==============================================================


def find_key_recursive(
    obj: Any,
    target_key: str,
):
    """
    Recursively search a nested YAML structure.

    This allows the script to work whether the retargeting
    configuration fields are at the YAML root or nested inside
    another dictionary.
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


# ==============================================================
# Pinocchio frame utilities
# ==============================================================


def find_frame_id_by_name(
    model: pin.Model,
    frame_name: str,
) -> int:
    """
    Find a Pinocchio frame by exact name.
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

        print()
        print(
            "[ERROR] Cannot find frame:"
        )

        print(
            f"    {frame_name}"
        )

        print()
        print(
            "Available frames containing "
            "'retarget' or similar:"
        )

        for frame_id, frame in enumerate(
            model.frames
        ):

            if (
                "retarget" in frame.name.lower()
                or
                frame_name.lower()
                in frame.name.lower()
            ):

                print(
                    f"    [{frame_id:3d}] "
                    f"{frame.name}"
                )

        raise RuntimeError(
            f"Cannot find Pinocchio frame: "
            f"{frame_name}"
        )

    if len(matches) > 1:

        print(
            "[WARNING] Multiple frames named "
            f"'{frame_name}'. "
            f"Using frame ID {matches[0]}."
        )

    return matches[0]


def get_frame_position(
    model: pin.Model,
    data: pin.Data,
    frame_name: str,
) -> tuple[int, np.ndarray]:
    """
    Return world-frame position of a Pinocchio frame.
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
            "ORCA URDF path."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "dex-retargeting VECTOR YAML."
        ),
    )

    args = parser.parse_args()


    # ----------------------------------------------------------
    # Input paths
    # ----------------------------------------------------------

    urdf_path = (
        Path(args.urdf)
        .expanduser()
        .resolve()
    )

    config_path = (
        Path(args.config)
        .expanduser()
        .resolve()
    )


    if not urdf_path.exists():

        raise FileNotFoundError(
            f"URDF does not exist:\n"
            f"{urdf_path}"
        )


    if not config_path.exists():

        raise FileNotFoundError(
            f"Config does not exist:\n"
            f"{config_path}"
        )


    # ----------------------------------------------------------
    # Load retargeting configuration
    # ----------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )


    origin_link_names = find_key_recursive(
        config,
        "target_origin_link_names",
    )

    task_link_names = find_key_recursive(
        config,
        "target_task_link_names",
    )


    if origin_link_names is None:

        raise KeyError(
            "Cannot find "
            "'target_origin_link_names' "
            "in retargeting YAML."
        )


    if task_link_names is None:

        raise KeyError(
            "Cannot find "
            "'target_task_link_names' "
            "in retargeting YAML."
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
            "Origin/task vector count mismatch:\n"
            f"origin={len(origin_link_names)}\n"
            f"task={len(task_link_names)}"
        )


    number_of_vectors = len(
        origin_link_names
    )


    # ----------------------------------------------------------
    # Build Pinocchio model
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
        f"Pinocchio joints : {model.njoints}"
    )

    print(
        f"Pinocchio frames : {len(model.frames)}"
    )

    print(
        f"Vector count     : {number_of_vectors}"
    )

    print()


    # ----------------------------------------------------------
    # q = 0
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
    # Forward kinematics
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
    # Vector mapping
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


        vector_length = float(
            np.linalg.norm(
                robot_vector
            )
        )


        robot_vectors.append(
            robot_vector
        )

        robot_lengths.append(
            vector_length
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
            "       origin="
            f"[{origin_position[0]:+.6f}, "
            f"{origin_position[1]:+.6f}, "
            f"{origin_position[2]:+.6f}]"
        )

        print(
            "       task  ="
            f"[{task_position[0]:+.6f}, "
            f"{task_position[1]:+.6f}, "
            f"{task_position[2]:+.6f}]"
        )

        print(
            "       v="
            f"[{robot_vector[0]:+.6f}, "
            f"{robot_vector[1]:+.6f}, "
            f"{robot_vector[2]:+.6f}]  "
            f"L={vector_length:.6f}"
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
    # Compact output matrix
    # ----------------------------------------------------------

    np.set_printoptions(
        precision=6,
        suppress=True,
    )


    print(
        "=" * 80
    )

    print(
        f"Robot vector matrix B "
        f"({number_of_vectors} x 3)"
    )

    print(
        "=" * 80
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


    for vector_index, vector_length in enumerate(
        robot_lengths
    ):

        print(
            f"RV{vector_index:02d}: "
            f"{vector_length:.6f}"
        )


    print()


    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

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
        "Mean vector length : "
        f"{robot_lengths.mean():.6f}"
    )

    print(
        "Min vector length  : "
        f"{robot_lengths.min():.6f}"
    )

    print(
        "Max vector length  : "
        f"{robot_lengths.max():.6f}"
    )

    print()

    print(
        "DONE."
    )


if __name__ == "__main__":
    main()
