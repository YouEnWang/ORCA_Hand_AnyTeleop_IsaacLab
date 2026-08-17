#!/usr/bin/env python3

"""
calibrate_orca_vector_alignment.py

Estimate the rigid coordinate-frame rotation and global scale
from human hand reference vectors to ORCA Hand reference vectors.

Pipeline
--------
Step 3:
    Human open-hand vectors
        H_i = p_task_i - p_origin_i

Step 4:
    ORCA q=0 robot vectors
        B_i = p_task_i - p_origin_i

We solve:

    B_i ~= scale * R * H_i

For row-vector representation used in NumPy:

    B ~= scale * H @ R.T

Important
---------
- Human vectors are already relative vectors, so translation
  should NOT be estimated.
- We do NOT center H or B like ordinary point-cloud Kabsch.
- Thumb vectors are excluded from calibration by default,
  because ORCA thumb morphology is substantially different
  from the four non-thumb fingers.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import yaml


# ==============================================================
# Constants
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


# Exclude thumb for global palm/frame calibration.
DEFAULT_CALIBRATION_INDICES = [
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
    Parse Step-3 [HVEC] diagnostics.

    Returns
    -------
    frames : list[dict]
        Each element contains:

        {
            "time": float,
            "vectors": np.ndarray shape (10,3)
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
    Parse Step-4 ORCA robot-vector diagnostic log.

    Expected Step-4 format:

        [RV00] origin_link -> task_link
               origin_frame_id=...
               origin=[...]
               task=[...]
               v=[x, y, z]  L=...

    Important:
    The [RVxx] label and v=[...] are on DIFFERENT lines,
    so this parser keeps track of the latest RV index.

    Returns
    -------
    robot_vectors : np.ndarray, shape (10, 3)
    """

    # ----------------------------------------------------------
    # Regex for:
    #
    # [RV00]
    # [RV01]
    # ...
    # ----------------------------------------------------------

    rv_index_regex = re.compile(
        r"\[RV(\d+)\]"
    )


    # ----------------------------------------------------------
    # Regex for:
    #
    # v=[-0.035445, +0.052595, -0.075095]
    # ----------------------------------------------------------

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

    # Only parse vectors after entering the actual
    # "ORCA robot vectors at q = 0" section.
    #
    # This avoids accidentally using the earlier
    # "Robot reference-vector mapping" section.
    inside_vector_section = False


    with open(
        log_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line in file:

            # --------------------------------------------------
            # Wait until the actual FK vector section starts
            # --------------------------------------------------

            if (
                "ORCA robot vectors at q = 0"
                in line
            ):

                inside_vector_section = True
                current_index = None
                continue


            if not inside_vector_section:
                continue


            # --------------------------------------------------
            # Stop before compact matrix section
            # --------------------------------------------------

            if (
                "Robot vector matrix B"
                in line
            ):

                break


            # --------------------------------------------------
            # Read latest RV index
            # --------------------------------------------------

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


            # --------------------------------------------------
            # Read vector belonging to current RV
            # --------------------------------------------------

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


                # Prevent another line from accidentally
                # overwriting the same RV entry.
                current_index = None


    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    if len(vectors) != 10:

        print()
        print(
            "[ERROR] Parsed robot-vector indices:"
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


    # ----------------------------------------------------------
    # Convert dict -> 10 x 3 matrix
    # ----------------------------------------------------------

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
# Rotation + scale estimation
# ==============================================================

def estimate_rotation_and_scale(
    human_vectors: np.ndarray,
    robot_vectors: np.ndarray,
):

    """
    Solve:

        B ~= s * H @ R.T

    where:
        H : N x 3 human vectors
        B : N x 3 robot vectors

    No translation and no centering are used.

    Returns
    -------
    R : 3x3
        Column-vector convention:

            b = s * R @ h

    scale : float
    """

    if human_vectors.shape != robot_vectors.shape:

        raise ValueError(
            "Human and robot matrices "
            "must have identical shape."
        )


    # ----------------------------------------------------------
    # Orthogonal Procrustes
    #
    # For row vectors:
    #
    #     H @ Q ~= B
    #
    # and:
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


    # Enforce proper rotation:
    #
    # det(R) = +1
    #
    # No reflection allowed.
    if np.linalg.det(
        preliminary_Q
    ) < 0.0:

        correction[
            -1,
            -1,
        ] = -1.0


    Q = (
        U
        @ correction
        @ Vt
    )


    # Column-vector rotation matrix.
    R = Q.T


    # ----------------------------------------------------------
    # Global least-squares scale
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
# Diagnostics
# ==============================================================

def vector_angle_deg(
    a: np.ndarray,
    b: np.ndarray,
):

    norm_a = float(
        np.linalg.norm(a)
    )

    norm_b = float(
        np.linalg.norm(b)
    )


    if (
        norm_a <= 1e-12
        or norm_b <= 1e-12
    ):

        return float("nan")


    cosine = float(
        np.dot(
            a,
            b,
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
            "Step-3 human-vector diagnostic log."
        ),
    )


    parser.add_argument(
        "--robot-log",
        required=True,
        type=str,
        help=(
            "Step-4 fixed-mapping robot-vector log."
        ),
    )


    parser.add_argument(
        "--human-start",
        type=float,
        default=2.0,
        help=(
            "Start time of stable human open-hand "
            "window in seconds."
        ),
    )


    parser.add_argument(
        "--human-end",
        type=float,
        default=5.0,
        help=(
            "End time of stable human open-hand "
            "window in seconds."
        ),
    )


    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help=(
            "Output YAML path."
        ),
    )


    args = parser.parse_args()


    human_log = Path(
        args.human_log
    ).expanduser().resolve()


    robot_log = Path(
        args.robot_log
    ).expanduser().resolve()


    output_path = Path(
        args.output
    ).expanduser().resolve()


    if not human_log.exists():

        raise FileNotFoundError(
            human_log
        )


    if not robot_log.exists():

        raise FileNotFoundError(
            robot_log
        )


    # ----------------------------------------------------------
    # Load human frames
    # ----------------------------------------------------------

    frames = parse_human_log(
        human_log
    )


    if len(frames) == 0:

        raise RuntimeError(
            "No complete HVEC frames "
            "were parsed."
        )


    selected_frames = [
        frame
        for frame in frames
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
            "No human frames fall inside "
            f"[{args.human_start}, "
            f"{args.human_end}] s."
        )


    human_stack = np.stack(
        [
            frame["vectors"]
            for frame in selected_frames
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
    # Load robot q=0 vectors
    # ----------------------------------------------------------

    robot_vectors = parse_robot_log(
        robot_log
    )


    calibration_indices = np.array(
        DEFAULT_CALIBRATION_INDICES,
        dtype=np.int64,
    )


    H_selected = human_mean[
        calibration_indices
    ]


    B_selected = robot_vectors[
        calibration_indices
    ]


    # ----------------------------------------------------------
    # Estimate alignment
    # ----------------------------------------------------------

    (
        R,
        scale,
        singular_values,
    ) = estimate_rotation_and_scale(
        H_selected,
        B_selected,
    )


    # Row-vector form:
    #
    # b = s * h @ R.T
    predicted_all = (
        scale
        *
        human_mean
        @ R.T
    )


    residual_vectors = (
        predicted_all
        -
        robot_vectors
    )


    residual_norms = (
        np.linalg.norm(
            residual_vectors,
            axis=1,
        )
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


    all_rmse = float(
        np.sqrt(
            np.mean(
                residual_norms
                ** 2
            )
        )
    )


    # ----------------------------------------------------------
    # Print input statistics
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
        f"Open window: "
        f"{args.human_start:.2f} "
        f"to "
        f"{args.human_end:.2f} s"
    )

    print(
        f"Human frames used: "
        f"{len(selected_frames)}"
    )

    print()


    print(
        "Calibration indices:"
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
        "Mean human OPEN vectors H"
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
        "Human vector per-axis standard deviation"
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
    # Alignment results
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
        "Rotation matrix R "
        "(column-vector convention):"
    )

    print()

    print(
        R
    )

    print()


    print(
        "Interpretation:"
    )

    print(
        "    v_orca = "
        "scale * R @ v_human"
    )

    print()

    print(
        "For row-vector NumPy arrays:"
    )

    print(
        "    V_orca = "
        "scale * V_human @ R.T"
    )

    print()


    print(
        f"det(R) = "
        f"{np.linalg.det(R):.8f}"
    )


    print(
        f"scale  = "
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
    # Per-vector diagnostics
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


    for index in range(10):

        human_vector = (
            human_mean[index]
        )

        predicted = (
            predicted_all[index]
        )

        robot = (
            robot_vectors[index]
        )

        residual = float(
            residual_norms[index]
        )

        angle = vector_angle_deg(
            predicted,
            robot,
        )


        used_marker = (
            "USED"
            if index in DEFAULT_CALIBRATION_INDICES
            else "CHECK_ONLY"
        )


        print(
            f"[{index:02d}] "
            f"{VECTOR_NAMES[index]:<20} "
            f"{used_marker}"
        )

        print(
            "     human     = "
            f"[{human_vector[0]:+.6f}, "
            f"{human_vector[1]:+.6f}, "
            f"{human_vector[2]:+.6f}]"
        )

        print(
            "     predicted = "
            f"[{predicted[0]:+.6f}, "
            f"{predicted[1]:+.6f}, "
            f"{predicted[2]:+.6f}]"
        )

        print(
            "     robot     = "
            f"[{robot[0]:+.6f}, "
            f"{robot[1]:+.6f}, "
            f"{robot[2]:+.6f}]"
        )

        print(
            f"     residual  = "
            f"{residual * 1000.0:.3f} mm"
        )

        print(
            f"     angle_err = "
            f"{angle:.3f} deg"
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
        f"{all_rmse * 1000.0:.3f} mm"
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

        "method": (
            "non_thumb_orthogonal_procrustes"
        ),

        "human_open_window_seconds": {
            "start": float(
                args.human_start
            ),
            "end": float(
                args.human_end
            ),
            "frames_used": int(
                len(selected_frames)
            ),
        },

        "selected_vector_indices": [
            int(index)
            for index
            in calibration_indices
        ],

        "selected_vector_names": [
            VECTOR_NAMES[
                int(index)
            ]
            for index
            in calibration_indices
        ],

        "human_to_orca_rotation": [
            [
                float(value)
                for value in row
            ]
            for row in R
        ],

        "scale": float(
            scale
        ),

        "convention": {
            "column_vector": (
                "v_orca = "
                "scale * R @ v_human"
            ),
            "row_vector_numpy": (
                "V_orca = "
                "scale * V_human @ R.T"
            ),
        },

        "diagnostics": {

            "det_rotation": float(
                np.linalg.det(R)
            ),

            "selected_rmse_m": float(
                selected_rmse
            ),

            "selected_rmse_mm": float(
                selected_rmse
                * 1000.0
            ),

            "all_vector_rmse_m": float(
                all_rmse
            ),

            "all_vector_rmse_mm": float(
                all_rmse
                * 1000.0
            ),

            "per_vector_residual_mm": {
                VECTOR_NAMES[index]:
                    float(
                        residual_norms[index]
                        * 1000.0
                    )
                for index in range(10)
            },
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
        f"Saved alignment YAML:"
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
