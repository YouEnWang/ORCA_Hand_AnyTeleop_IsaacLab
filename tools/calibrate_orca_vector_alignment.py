#!/usr/bin/env python3

"""
calibrate_orca_vector_alignment.py

Estimate Human -> ORCA vector-frame alignment.

Inputs
------
1. Human vector diagnostic log produced by
   orca_anyteleop_server.py:

       [HVEC] t=...
       [HV00] ... v=[x,y,z]
       ...
       [HV09] ... v=[x,y,z]

2. ORCA q=0 robot vector diagnostic log produced by
   inspect_orca_robot_vectors.py:

       [RV00] ...
              ...
              v=[x,y,z] L=...

Goal
----
Find rotation R and scale s such that:

    column-vector convention:

        v_orca ~= s * R @ v_human

    row-vector NumPy convention:

        V_orca ~= s * V_human @ R.T

Important
---------
Human and robot quantities are already relative vectors
(task position - origin position), so translation is NOT
estimated and the vectors are NOT mean-centered.

The global calibration uses the 8 non-thumb vectors:

    1,2,3,4  : fingertip vectors
    6,7,8,9  : intermediate/PIP vectors

Thumb vectors 0 and 5 are evaluated but do not influence
the global alignment.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import yaml


# ==============================================================
# Vector semantics
# ==============================================================

VECTOR_NAMES = {
    0: "thumb_tip",
    1: "index_tip",
    2: "middle_tip",
    3: "ring_tip",
    4: "pinky_tip",

    5: "thumb_intermediate",
    6: "index_intermediate",
    7: "middle_intermediate",
    8: "ring_intermediate",
    9: "pinky_intermediate",
}


# Do not use thumb to determine the global palm-frame alignment.
CALIBRATION_INDICES = [
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
]


FLOAT_PATTERN = (
    r"[-+]?"
    r"(?:\d+\.\d+|\d+|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


# ==============================================================
# Human log parser
# ==============================================================

def parse_human_log(
    log_path: Path,
):
    """
    Parse [HVEC] blocks from the Step-3 / realtime server log.

    Returns
    -------
    frames : list of dict

        {
            "time": float,
            "vectors": ndarray shape (10, 3)
        }
    """

    time_regex = re.compile(
        r"\[HVEC\]\s+t=\s*("
        + FLOAT_PATTERN
        + r")s"
    )

    vector_regex = re.compile(
        r"\[HV(\d+)\].*?"
        r"v=\[\s*("
        + FLOAT_PATTERN
        + r")\s*,\s*("
        + FLOAT_PATTERN
        + r")\s*,\s*("
        + FLOAT_PATTERN
        + r")\s*\]"
    )


    frames = []

    current_time = None
    current_vectors = {}


    def flush_current():

        nonlocal current_time
        nonlocal current_vectors

        if current_time is None:
            return

        if len(current_vectors) != 10:
            return

        matrix = np.zeros(
            (10, 3),
            dtype=np.float64,
        )

        for index in range(10):

            if index not in current_vectors:
                return

            matrix[index] = (
                current_vectors[index]
            )

        frames.append(
            {
                "time": current_time,
                "vectors": matrix,
            }
        )


    with open(
        log_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line in file:

            time_match = (
                time_regex.search(line)
            )

            if time_match:

                flush_current()

                current_time = float(
                    time_match.group(1)
                )

                current_vectors = {}

                continue


            if current_time is None:
                continue


            vector_match = (
                vector_regex.search(line)
            )

            if vector_match:

                vector_index = int(
                    vector_match.group(1)
                )

                vector = np.array(
                    [
                        float(vector_match.group(2)),
                        float(vector_match.group(3)),
                        float(vector_match.group(4)),
                    ],
                    dtype=np.float64,
                )

                current_vectors[
                    vector_index
                ] = vector


    flush_current()

    return frames


# ==============================================================
# Robot log parser
# ==============================================================

def parse_robot_log(
    log_path: Path,
):
    """
    Parse the multi-line RV blocks produced by
    inspect_orca_robot_vectors.py.

    Expected format:

        ORCA robot vectors at q = 0

        [RV00] palm -> task
               origin=...
               task=...
               v=[x, y, z] L=...

    The [RVxx] line and v=[...] line are separate,
    therefore the parser keeps the latest RV index.
    """

    rv_index_regex = re.compile(
        r"\[RV(\d+)\]"
    )

    vector_regex = re.compile(
        r"^\s*v=\[\s*("
        + FLOAT_PATTERN
        + r")\s*,\s*("
        + FLOAT_PATTERN
        + r")\s*,\s*("
        + FLOAT_PATTERN
        + r")\s*\]"
    )


    vectors = {}

    current_index = None

    inside_vector_section = False


    with open(
        log_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line in file:

            # Start only after the actual numerical vector section.
            if (
                "ORCA robot vectors at q = 0"
                in line
            ):

                inside_vector_section = True

                current_index = None

                continue


            if not inside_vector_section:
                continue


            # Stop before the compact matrix dump.
            if (
                "Robot vector matrix B"
                in line
            ):

                break


            index_match = (
                rv_index_regex.search(
                    line
                )
            )

            if index_match:

                current_index = int(
                    index_match.group(1)
                )

                continue


            if current_index is None:
                continue


            vector_match = (
                vector_regex.search(
                    line
                )
            )


            if vector_match:

                vector = np.array(
                    [
                        float(
                            vector_match.group(1)
                        ),
                        float(
                            vector_match.group(2)
                        ),
                        float(
                            vector_match.group(3)
                        ),
                    ],
                    dtype=np.float64,
                )


                vectors[
                    current_index
                ] = vector


                current_index = None


    if len(vectors) != 10:

        print()
        print(
            "[ERROR] Parsed RV indices:"
        )

        print(
            sorted(
                vectors.keys()
            )
        )

        print()

        raise RuntimeError(
            "Expected 10 robot vectors, "
            f"but parsed {len(vectors)}."
        )


    matrix = np.zeros(
        (10, 3),
        dtype=np.float64,
    )


    for index in range(10):

        if index not in vectors:

            raise RuntimeError(
                f"Missing RV{index:02d}."
            )

        matrix[
            index
        ] = vectors[
            index
        ]


    return matrix


# ==============================================================
# Alignment estimation
# ==============================================================

def estimate_rotation_and_scale(
    human_vectors: np.ndarray,
    robot_vectors: np.ndarray,
):
    """
    Solve:

        B ~= s * H @ R.T

    using uncentered orthogonal Procrustes.

    Returns
    -------
    R : ndarray (3, 3)

        Column-vector convention:

            b = s * R @ h

    scale : float
    singular_values : ndarray
    """

    if (
        human_vectors.shape
        != robot_vectors.shape
    ):

        raise ValueError(
            "Human and robot vector matrices "
            "must have identical shapes."
        )


    # ----------------------------------------------------------
    # Row-vector orthogonal Procrustes:
    #
    #     H @ Q ~= B
    #
    # with:
    #
    #     Q = R.T
    # ----------------------------------------------------------

    covariance = (
        human_vectors.T
        @ robot_vectors
    )


    U, singular_values, Vt = (
        np.linalg.svd(
            covariance
        )
    )


    correction = np.eye(
        3,
        dtype=np.float64,
    )


    preliminary_Q = (
        U @ Vt
    )


    # Proper rotation only.
    # Reflection is forbidden.
    if (
        np.linalg.det(
            preliminary_Q
        )
        < 0.0
    ):

        correction[
            -1,
            -1,
        ] = -1.0


    Q = (
        U
        @ correction
        @ Vt
    )


    # Column-vector rotation.
    R = Q.T


    # ----------------------------------------------------------
    # Least-squares global scale
    # ----------------------------------------------------------

    human_rotated = (
        human_vectors
        @ Q
    )


    numerator = float(
        np.sum(
            robot_vectors
            * human_rotated
        )
    )


    denominator = float(
        np.sum(
            human_rotated
            * human_rotated
        )
    )


    if denominator <= 1e-12:

        raise RuntimeError(
            "Degenerate human vector matrix."
        )


    scale = (
        numerator
        /
        denominator
    )


    return (
        R,
        scale,
        singular_values,
    )


# ==============================================================
# Diagnostic utilities
# ==============================================================

def vector_angle_deg(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
):
    """
    Angle between two 3-D vectors in degrees.
    """

    norm_a = float(
        np.linalg.norm(
            vector_a
        )
    )

    norm_b = float(
        np.linalg.norm(
            vector_b
        )
    )


    if (
        norm_a <= 1e-12
        or
        norm_b <= 1e-12
    ):

        return float("nan")


    cosine = float(
        np.dot(
            vector_a,
            vector_b,
        )
        /
        (
            norm_a
            *
            norm_b
        )
    )


    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )


    return math.degrees(
        math.acos(
            cosine
        )
    )


# ==============================================================
# Main
# ==============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--human-log",
        required=True,
        type=str,
        help=(
            "Human HVEC diagnostic log."
        ),
    )


    parser.add_argument(
        "--robot-log",
        required=True,
        type=str,
        help=(
            "Robot RV q=0 diagnostic log."
        ),
    )


    parser.add_argument(
        "--human-start",
        type=float,
        default=2.0,
        help=(
            "Start of stable human OPEN window."
        ),
    )


    parser.add_argument(
        "--human-end",
        type=float,
        default=5.0,
        help=(
            "End of stable human OPEN window."
        ),
    )


    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help=(
            "Output alignment YAML."
        ),
    )


    args = parser.parse_args()


    human_log = (
        Path(args.human_log)
        .expanduser()
        .resolve()
    )

    robot_log = (
        Path(args.robot_log)
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(args.output)
        .expanduser()
        .resolve()
    )


    if not human_log.exists():

        raise FileNotFoundError(
            human_log
        )


    if not robot_log.exists():

        raise FileNotFoundError(
            robot_log
        )


    # ----------------------------------------------------------
    # Parse Human vectors
    # ----------------------------------------------------------

    human_frames = parse_human_log(
        human_log
    )


    if len(human_frames) == 0:

        raise RuntimeError(
            "No complete Human HVEC frames "
            "were parsed."
        )


    selected_frames = [
        frame
        for frame
        in human_frames
        if (
            frame["time"]
            >= args.human_start
            and
            frame["time"]
            <= args.human_end
        )
    ]


    if len(selected_frames) == 0:

        raise RuntimeError(
            "No Human frames in requested "
            f"OPEN interval "
            f"[{args.human_start}, "
            f"{args.human_end}] s."
        )


    human_stack = np.stack(
        [
            frame["vectors"]
            for frame
            in selected_frames
        ],
        axis=0,
    )


    human_mean = np.mean(
        human_stack,
        axis=0,
    )


    human_std = np.std(
        human_stack,
        axis=0,
    )


    # ----------------------------------------------------------
    # Parse Robot vectors
    # ----------------------------------------------------------

    robot_vectors = parse_robot_log(
        robot_log
    )


    calibration_indices = np.asarray(
        CALIBRATION_INDICES,
        dtype=np.int64,
    )


    H_selected = human_mean[
        calibration_indices
    ]


    B_selected = robot_vectors[
        calibration_indices
    ]


    # ----------------------------------------------------------
    # Solve R and scale
    # ----------------------------------------------------------

    (
        rotation,
        scale,
        singular_values,
    ) = estimate_rotation_and_scale(
        H_selected,
        B_selected,
    )


    # ----------------------------------------------------------
    # Predict all 10 vectors
    #
    # Row-vector convention:
    #
    # B_pred = scale * H @ R.T
    # ----------------------------------------------------------

    predicted_vectors = (
        scale
        *
        human_mean
        @ rotation.T
    )


    residual_vectors = (
        predicted_vectors
        -
        robot_vectors
    )


    residual_norms = np.linalg.norm(
        residual_vectors,
        axis=1,
    )


    selected_residuals = (
        residual_norms[
            calibration_indices
        ]
    )


    selected_rmse = float(
        np.sqrt(
            np.mean(
                selected_residuals
                ** 2
            )
        )
    )


    all_vector_rmse = float(
        np.sqrt(
            np.mean(
                residual_norms
                ** 2
            )
        )
    )


    # ----------------------------------------------------------
    # Pretty output
    # ----------------------------------------------------------

    np.set_printoptions(
        precision=6,
        suppress=True,
    )


    print(
        "=" * 80
    )

    print(
        "Human -> ORCA Vector Alignment Calibration"
    )

    print(
        "=" * 80
    )

    print(
        f"Human log : {human_log}"
    )

    print(
        f"Robot log : {robot_log}"
    )

    print(
        f"OPEN window : "
        f"{args.human_start:.2f} "
        f"to "
        f"{args.human_end:.2f} s"
    )

    print(
        f"Human frames used : "
        f"{len(selected_frames)}"
    )

    print()


    print(
        "Calibration vectors:"
    )

    for index in calibration_indices:

        print(
            f"  {index:02d}: "
            f"{VECTOR_NAMES[int(index)]}"
        )

    print()


    print(
        "=" * 80
    )

    print(
        "Mean Human OPEN vectors H"
    )

    print(
        "=" * 80
    )

    print(
        human_mean
    )

    print()


    print(
        "=" * 80
    )

    print(
        "Human OPEN vector standard deviation"
    )

    print(
        "=" * 80
    )

    print(
        human_std
    )

    print()


    print(
        "=" * 80
    )

    print(
        "Robot q=0 vectors B"
    )

    print(
        "=" * 80
    )

    print(
        robot_vectors
    )

    print()


    # ----------------------------------------------------------
    # Result
    # ----------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "Estimated Human -> ORCA alignment"
    )

    print(
        "=" * 80
    )

    print(
        "Rotation matrix R:"
    )

    print()

    print(
        rotation
    )

    print()

    print(
        "Column-vector convention:"
    )

    print(
        "    v_orca = "
        "scale * R @ v_human"
    )

    print()

    print(
        "Row-vector NumPy convention:"
    )

    print(
        "    V_orca = "
        "scale * V_human @ R.T"
    )

    print()


    determinant = float(
        np.linalg.det(
            rotation
        )
    )


    orthogonality_error = float(
        np.max(
            np.abs(
                rotation
                @ rotation.T
                -
                np.eye(3)
            )
        )
    )


    print(
        f"det(R) = "
        f"{determinant:.8f}"
    )

    print(
        f"Orthogonality error = "
        f"{orthogonality_error:.8e}"
    )

    print(
        f"scale = "
        f"{scale:.8f}"
    )

    print(
        "SVD singular values ="
    )

    print(
        singular_values
    )

    print()


    # ----------------------------------------------------------
    # Per-vector fit
    # ----------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "Per-vector fit diagnostics"
    )

    print(
        "=" * 80
    )


    per_vector_residual_mm = {}
    per_vector_angle_deg = {}


    for index in range(10):

        used = (
            index
            in CALIBRATION_INDICES
        )

        status = (
            "USED"
            if used
            else "CHECK_ONLY"
        )


        human_vector = (
            human_mean[index]
        )

        predicted_vector = (
            predicted_vectors[index]
        )

        robot_vector = (
            robot_vectors[index]
        )


        residual_mm = float(
            residual_norms[index]
            * 1000.0
        )


        angle_error = vector_angle_deg(
            predicted_vector,
            robot_vector,
        )


        per_vector_residual_mm[
            VECTOR_NAMES[index]
        ] = residual_mm


        per_vector_angle_deg[
            VECTOR_NAMES[index]
        ] = float(
            angle_error
        )


        print(
            f"[{index:02d}] "
            f"{VECTOR_NAMES[index]:<20} "
            f"{status}"
        )

        print(
            "     human     = "
            f"[{human_vector[0]:+.6f}, "
            f"{human_vector[1]:+.6f}, "
            f"{human_vector[2]:+.6f}]"
        )

        print(
            "     predicted = "
            f"[{predicted_vector[0]:+.6f}, "
            f"{predicted_vector[1]:+.6f}, "
            f"{predicted_vector[2]:+.6f}]"
        )

        print(
            "     robot     = "
            f"[{robot_vector[0]:+.6f}, "
            f"{robot_vector[1]:+.6f}, "
            f"{robot_vector[2]:+.6f}]"
        )

        print(
            f"     residual  = "
            f"{residual_mm:.3f} mm"
        )

        print(
            f"     angle_err = "
            f"{angle_error:.3f} deg"
        )

        print()


    print(
        "=" * 80
    )

    print(
        "Fit summary"
    )

    print(
        "=" * 80
    )

    print(
        "Selected non-thumb RMSE : "
        f"{selected_rmse * 1000.0:.3f} mm"
    )

    print(
        "All-vector RMSE         : "
        f"{all_vector_rmse * 1000.0:.3f} mm"
    )

    print()


    # ----------------------------------------------------------
    # Save YAML
    # ----------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_data = {

        "method":
            "non_thumb_uncentered_orthogonal_procrustes",

        "human_open_window_seconds": {
            "start":
                float(args.human_start),

            "end":
                float(args.human_end),

            "frames_used":
                int(len(selected_frames)),
        },

        "selected_vector_indices": [
            int(index)
            for index
            in calibration_indices
        ],

        "selected_vector_names": [
            VECTOR_NAMES[int(index)]
            for index
            in calibration_indices
        ],

        "human_to_orca_rotation": [
            [
                float(value)
                for value
                in row
            ]
            for row
            in rotation
        ],

        "scale":
            float(scale),

        "convention": {

            "column_vector":
                "v_orca = scale * R @ v_human",

            "row_vector_numpy":
                "V_orca = scale * V_human @ R.T",
        },

        "diagnostics": {

            "det_rotation":
                determinant,

            "orthogonality_error":
                orthogonality_error,

            "selected_rmse_m":
                float(selected_rmse),

            "selected_rmse_mm":
                float(
                    selected_rmse
                    * 1000.0
                ),

            "all_vector_rmse_m":
                float(all_vector_rmse),

            "all_vector_rmse_mm":
                float(
                    all_vector_rmse
                    * 1000.0
                ),

            "per_vector_residual_mm":
                per_vector_residual_mm,

            "per_vector_angle_error_deg":
                per_vector_angle_deg,
        },
    }


    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            output_data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


    print(
        "Saved alignment YAML:"
    )

    print(
        f"    {output_path}"
    )

    print()

    print(
        "DONE."
    )


if __name__ == "__main__":
    main()
