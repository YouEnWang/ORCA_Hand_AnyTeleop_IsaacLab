#!/usr/bin/env python3

"""
Gate 3B-3 — Static Mapping Validation

Analyze the Gate 3B offline replay results for:

    static_open
    static_half
    static_pinch

Each condition contains:
    R1, R2, R3
    alpha = 0.2  (canonical LPF)
    alpha = 1.0  (LPF bypass)

The script analyzes three command layers:

    q_optimizer_target
        16 optimized ORCA finger joints.

    q_robot_prefilter
        Full 17-DoF robot q before output LPF.

    q_robot_output
        Full 17-DoF returned command after LPF.

Outputs
-------
results/gate3b/static/

    gate3b_static_per_run.csv
    gate3b_static_per_joint.csv
    gate3b_static_condition_summary.csv
    gate3b_static_pose_contrasts.csv
    gate3b_static_between_run_repeatability.csv
    gate3b_static_validation.csv
    gate3b_static_summary.json

    static_pose_joint_means_alpha02.png
    static_pose_joint_means_alpha10.png
    static_pose_joint_std_alpha02.png
    static_pose_joint_std_alpha10.png
    static_frame_delta_rms_alpha02.png
    static_frame_delta_rms_alpha10.png
    static_filter_suppression.png

Notes
-----
1. Full-run statistics are preserved.
2. "steady" statistics discard the first --trim-start-s seconds.
3. Invalid frames are never interpolated.
4. Statistics are computed only inside contiguous valid data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    "static_open",
    "static_half",
    "static_pinch",
]

ROUNDS = ["r1", "r2", "r3"]

MODES = {
    "alpha02": 0.2,
    "alpha10": 1.0,
}

LAYERS = {
    "optimizer": "q_optimizer_target",
    "prefilter": "q_robot_prefilter",
    "output": "q_robot_output",
}


# ------------------------------------------------------------
# ORCA display names
# ------------------------------------------------------------

def short_joint_name(name: str) -> str:
    """
    Produce readable labels while preserving ORCA joint identity.
    """

    mapping = {
        "R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e":
            "Wrist",

        "I-AP-R_d95d02d1_to_R-Carpals_8d1f1041":
            "Index ABD",
        "I-PP_bacbd481_to_I-AP-R_d95d02d1":
            "Index MCP",
        "I-FingerTipAssembly_ec49c16c_to_I-PP_bacbd481":
            "Index PIP",

        "M-AP_6ec59111_to_R-Carpals_8d1f1041":
            "Middle ABD",
        "M-PP_8660a1eb_to_M-AP_6ec59111":
            "Middle MCP",
        "M-FingerTipAssembly_424a8e75_to_M-PP_8660a1eb":
            "Middle PIP",

        "M-AP_e04a96f2_to_R-Carpals_8d1f1041":
            "Ring ABD",
        "M-PP_08efa608_to_M-AP_e04a96f2":
            "Ring MCP",
        "M-FingerTipAssembly_34afb748_to_M-PP_08efa608":
            "Ring PIP",

        "P-AP_f5e42b61_to_R-Carpals_8d1f1041":
            "Pinky ABD",
        "P-PP_1d411b9b_to_P-AP_f5e42b61":
            "Pinky MCP",
        "P-FingerTipAssembly_cd219176_to_P-PP_1d411b9b":
            "Pinky PIP",

        "T-TP-R_1c2b802d_to_R-Carpals_8d1f1041":
            "Thumb CMC",
        "R-T-AP_a9723101_to_T-TP-R_1c2b802d":
            "Thumb ABD",
        "T-PP_68395e98_to_R-T-AP_a9723101":
            "Thumb MCP",
        "T-DP_b7429e50_to_T-PP_68395e98":
            "Thumb DIP",
    }

    return mapping.get(name, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate 3B static mapping analysis"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/gate3b/replay"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gate3b/static"),
    )

    parser.add_argument(
        "--trim-start-s",
        type=float,
        default=2.0,
        help=(
            "Seconds removed from the beginning when computing "
            "steady-state statistics."
        ),
    )

    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_relative_time(rows):
    t = np.array(
        [
            float(r["capture_monotonic_s"])
            if r.get("capture_monotonic_s") is not None
            else np.nan
            for r in rows
        ],
        dtype=float,
    )

    finite = np.isfinite(t)

    if np.any(finite):
        t0 = t[np.where(finite)[0][0]]
        return t - t0

    # fallback
    return np.arange(len(rows), dtype=float) / 30.0


def get_layer_array(rows, key):
    """
    Return:
        q        : N x D array
        valid    : N bool
        names    : D list
    """

    if key == "q_optimizer_target":
        names_key = "target_joint_names"
    else:
        names_key = "robot_joint_names"

    names = None

    for r in rows:
        candidate = r.get(names_key)

        if candidate:
            names = list(candidate)
            break

    if names is None:
        raise RuntimeError(
            f"Could not find {names_key}"
        )

    dim = len(names)

    q = np.full(
        (len(rows), dim),
        np.nan,
        dtype=float,
    )

    valid = np.zeros(
        len(rows),
        dtype=bool,
    )

    for i, r in enumerate(rows):

        if not bool(r.get("tracking_valid", False)):
            continue

        x = r.get(key)

        if x is None:
            continue

        x = np.asarray(x, dtype=float)

        if x.shape != (dim,):
            raise ValueError(
                f"{key}: expected shape {(dim,)}, got {x.shape}"
            )

        if not np.all(np.isfinite(x)):
            continue

        q[i] = x
        valid[i] = True

    return q, valid, names


def contiguous_delta(q, valid, t):
    """
    Compute q[t] - q[t-1] only for truly consecutive valid frames.

    Returns:
        dq      M x D
        dt      M
    """

    idx = np.where(valid)[0]

    if len(idx) < 2:
        return (
            np.empty((0, q.shape[1])),
            np.empty((0,)),
        )

    dqs = []
    dts = []

    all_dt = np.diff(t[np.isfinite(t)])

    if len(all_dt):
        median_dt = np.nanmedian(all_dt)
    else:
        median_dt = 1.0 / 30.0

    max_gap = 2.5 * median_dt

    for a, b in zip(idx[:-1], idx[1:]):

        if b != a + 1:
            continue

        dt = t[b] - t[a]

        if not np.isfinite(dt):
            continue

        if dt <= 0 or dt > max_gap:
            continue

        dqs.append(q[b] - q[a])
        dts.append(dt)

    if not dqs:
        return (
            np.empty((0, q.shape[1])),
            np.empty((0,)),
        )

    return (
        np.asarray(dqs),
        np.asarray(dts),
    )


def safe_rms(x, axis=None):
    if np.size(x) == 0:
        return np.nan

    return np.sqrt(
        np.nanmean(
            np.square(x),
            axis=axis,
        )
    )


def compute_layer_stats(
    q,
    valid,
    t,
    names,
    trim_start_s,
):
    stats = []

    masks = {
        "full": valid,
        "steady": valid & (t >= trim_start_s),
    }

    for window_name, mask in masks.items():

        q_sel = q[mask]

        if len(q_sel) == 0:
            continue

        dq, dt = contiguous_delta(
            q,
            mask,
            t,
        )

        for j, joint_name in enumerate(names):

            values = q_sel[:, j]

            if len(dq):
                delta = dq[:, j]
            else:
                delta = np.array([], dtype=float)

            stats.append({
                "window": window_name,
                "joint_index": j,
                "joint_name": joint_name,
                "joint_short": short_joint_name(
                    joint_name
                ),

                "samples": int(
                    np.isfinite(values).sum()
                ),

                "mean_rad": float(
                    np.nanmean(values)
                ),

                "std_rad": float(
                    np.nanstd(values)
                ),

                "min_rad": float(
                    np.nanmin(values)
                ),

                "max_rad": float(
                    np.nanmax(values)
                ),

                "range_rad": float(
                    np.nanmax(values)
                    - np.nanmin(values)
                ),

                "mean_deg": float(
                    np.degrees(
                        np.nanmean(values)
                    )
                ),

                "std_deg": float(
                    np.degrees(
                        np.nanstd(values)
                    )
                ),

                "range_deg": float(
                    np.degrees(
                        np.nanmax(values)
                        - np.nanmin(values)
                    )
                ),

                "delta_rms_rad_per_frame":
                    float(safe_rms(delta)),

                "delta_rms_deg_per_frame":
                    float(
                        np.degrees(
                            safe_rms(delta)
                        )
                    ),

                "delta_p95_deg_per_frame":
                    float(
                        np.degrees(
                            np.nanpercentile(
                                np.abs(delta),
                                95,
                            )
                        )
                    )
                    if len(delta)
                    else np.nan,

                "delta_max_deg_per_frame":
                    float(
                        np.degrees(
                            np.nanmax(
                                np.abs(delta)
                            )
                        )
                    )
                    if len(delta)
                    else np.nan,
            })

    return stats


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def mean_vector_for_run(
    rows,
    layer_key,
    trim_start_s,
):
    q, valid, names = get_layer_array(
        rows,
        layer_key,
    )

    t = get_relative_time(rows)

    mask = valid & (
        t >= trim_start_s
    )

    if not np.any(mask):
        return None, names

    return (
        np.nanmean(q[mask], axis=0),
        names,
    )


def replay_validation(rows, mode):
    """
    alpha02:
        q_robot_output should reproduce Gate3A positions_rad.

    alpha10:
        q_robot_output should equal q_robot_prefilter.
    """

    errors = []

    for r in rows:

        if not r.get(
            "tracking_valid",
            False,
        ):
            continue

        if mode == "alpha02":
            a = r.get("q_gate3a_logged")
            b = r.get("q_robot_output")

        else:
            a = r.get("q_robot_prefilter")
            b = r.get("q_robot_output")

        if a is None or b is None:
            continue

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        errors.append(
            np.abs(a - b)
        )

    if not errors:
        return {
            "mean_abs_rad": np.nan,
            "max_abs_rad": np.nan,
        }

    e = np.asarray(errors)

    return {
        "mean_abs_rad":
            float(np.mean(e)),

        "max_abs_rad":
            float(np.max(e)),
    }


def make_grouped_bar(
    summary_rows,
    mode,
    metric,
    ylabel,
    output_path,
):
    """
    Plot the output-layer condition means.
    """

    selected = [
        r for r in summary_rows
        if r["mode"] == mode
        and r["layer"] == "output"
        and r["window"] == "steady"
    ]

    if not selected:
        return

    joint_names = []
    for r in selected:
        if r["joint_short"] not in joint_names:
            joint_names.append(
                r["joint_short"]
            )

    conditions = [
        "static_open",
        "static_half",
        "static_pinch",
    ]

    x = np.arange(
        len(joint_names)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    for k, condition in enumerate(
        conditions
    ):

        vals = []

        for joint in joint_names:

            row = next(
                (
                    z for z in selected
                    if z["condition"] == condition
                    and z["joint_short"] == joint
                ),
                None,
            )

            vals.append(
                np.nan
                if row is None
                else row[metric]
            )

        ax.bar(
            x + (k - 1) * width,
            vals,
            width,
            label=condition,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        joint_names,
        rotation=60,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Gate 3B Static Mapping — {mode}"
    )
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_joint_rows = []
    per_run_rows = []
    validation_rows = []

    loaded = {}

    # ============================================================
    # Load 18 expected static replay files
    # ============================================================

    for condition in CONDITIONS:
        for round_name in ROUNDS:
            for mode, alpha in MODES.items():

                path = (
                    input_dir
                    / f"{condition}_{round_name}_{mode}.jsonl"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        path
                    )

                rows = load_jsonl(path)

                loaded[
                    (condition, round_name, mode)
                ] = rows

                t = get_relative_time(rows)

                valid_count = sum(
                    bool(
                        r.get(
                            "tracking_valid",
                            False,
                        )
                    )
                    for r in rows
                )

                invalid_count = (
                    len(rows) - valid_count
                )

                validation = replay_validation(
                    rows,
                    mode,
                )

                validation_rows.append({
                    "condition": condition,
                    "round": round_name,
                    "mode": mode,
                    "alpha": alpha,
                    "frames": len(rows),
                    "valid": valid_count,
                    "invalid": invalid_count,
                    "replay_mean_abs_rad":
                        validation[
                            "mean_abs_rad"
                        ],
                    "replay_max_abs_rad":
                        validation[
                            "max_abs_rad"
                        ],
                    "replay_max_abs_deg":
                        float(
                            np.degrees(
                                validation[
                                    "max_abs_rad"
                                ]
                            )
                        ),
                })

                # --------------------------------------------
                # Analyze all three layers
                # --------------------------------------------

                for layer_name, layer_key in (
                    LAYERS.items()
                ):

                    q, valid, names = (
                        get_layer_array(
                            rows,
                            layer_key,
                        )
                    )

                    stats = compute_layer_stats(
                        q,
                        valid,
                        t,
                        names,
                        args.trim_start_s,
                    )

                    for stat in stats:
                        stat.update({
                            "condition":
                                condition,
                            "round":
                                round_name,
                            "mode":
                                mode,
                            "alpha":
                                alpha,
                            "layer":
                                layer_name,
                        })

                        per_joint_rows.append(
                            stat
                        )

                    # ----------------------------------------
                    # Per-run aggregate metric
                    # ----------------------------------------

                    mask = (
                        valid
                        & (
                            t
                            >= args.trim_start_s
                        )
                    )

                    dq, _ = contiguous_delta(
                        q,
                        mask,
                        t,
                    )

                    per_run_rows.append({
                        "condition":
                            condition,
                        "round":
                            round_name,
                        "mode":
                            mode,
                        "alpha":
                            alpha,
                        "layer":
                            layer_name,
                        "frames":
                            len(rows),
                        "valid_frames":
                            int(valid.sum()),
                        "steady_valid_frames":
                            int(mask.sum()),

                        "mean_joint_std_deg":
                            float(
                                np.nanmean(
                                    np.degrees(
                                        np.nanstd(
                                            q[mask],
                                            axis=0,
                                        )
                                    )
                                )
                            ),

                        "global_delta_rms_deg_per_frame":
                            float(
                                np.degrees(
                                    safe_rms(dq)
                                )
                            )
                            if len(dq)
                            else np.nan,
                    })

    # ============================================================
    # Condition summary
    # ============================================================

    condition_summary = []

    groups = defaultdict(list)

    for r in per_joint_rows:

        if r["window"] != "steady":
            continue

        key = (
            r["condition"],
            r["mode"],
            r["layer"],
            r["joint_index"],
            r["joint_name"],
            r["joint_short"],
        )

        groups[key].append(r)

    for key, rows in groups.items():

        (
            condition,
            mode,
            layer,
            joint_index,
            joint_name,
            joint_short,
        ) = key

        run_means = np.array(
            [
                r["mean_deg"]
                for r in rows
            ],
            dtype=float,
        )

        run_stds = np.array(
            [
                r["std_deg"]
                for r in rows
            ],
            dtype=float,
        )

        run_delta = np.array(
            [
                r[
                    "delta_rms_deg_per_frame"
                ]
                for r in rows
            ],
            dtype=float,
        )

        condition_summary.append({
            "condition":
                condition,

            "mode":
                mode,

            "alpha":
                MODES[mode],

            "layer":
                layer,

            "window":
                "steady",

            "joint_index":
                joint_index,

            "joint_name":
                joint_name,

            "joint_short":
                joint_short,

            "mean_of_run_means_deg":
                float(
                    np.nanmean(
                        run_means
                    )
                ),

            "between_run_std_deg":
                float(
                    np.nanstd(
                        run_means
                    )
                ),

            "mean_within_run_std_deg":
                float(
                    np.nanmean(
                        run_stds
                    )
                ),

            "mean_delta_rms_deg_per_frame":
                float(
                    np.nanmean(
                        run_delta
                    )
                ),
        })

    # ============================================================
    # Pose contrasts
    #
    # Half - Open
    # Pinch - Open
    # ============================================================

    contrast_rows = []

    for mode in MODES:
        for layer in LAYERS:

            lookup = {}

            for r in condition_summary:

                if (
                    r["mode"] == mode
                    and r["layer"] == layer
                ):
                    lookup[
                        (
                            r["condition"],
                            r["joint_name"],
                        )
                    ] = r

            joint_names = sorted(
                set(
                    joint
                    for condition, joint
                    in lookup.keys()
                )
            )

            for joint in joint_names:

                open_row = lookup.get(
                    ("static_open", joint)
                )

                if open_row is None:
                    continue

                for condition, label in [
                    (
                        "static_half",
                        "half_minus_open",
                    ),
                    (
                        "static_pinch",
                        "pinch_minus_open",
                    ),
                ]:

                    target = lookup.get(
                        (condition, joint)
                    )

                    if target is None:
                        continue

                    contrast_rows.append({
                        "mode":
                            mode,

                        "layer":
                            layer,

                        "contrast":
                            label,

                        "joint_name":
                            joint,

                        "joint_short":
                            target[
                                "joint_short"
                            ],

                        "delta_mean_deg":
                            (
                                target[
                                    "mean_of_run_means_deg"
                                ]
                                -
                                open_row[
                                    "mean_of_run_means_deg"
                                ]
                            ),
                    })

    # ============================================================
    # Between-run repeatability
    # ============================================================

    repeatability_rows = []

    for condition in CONDITIONS:
        for mode in MODES:
            for layer_name, layer_key in (
                LAYERS.items()
            ):

                vectors = {}

                names_ref = None

                for round_name in ROUNDS:

                    rows = loaded[
                        (
                            condition,
                            round_name,
                            mode,
                        )
                    ]

                    vec, names = (
                        mean_vector_for_run(
                            rows,
                            layer_key,
                            args.trim_start_s,
                        )
                    )

                    if vec is None:
                        continue

                    vectors[
                        round_name
                    ] = vec

                    names_ref = names

                for a, b in combinations(
                    sorted(vectors),
                    2,
                ):

                    diff = (
                        vectors[a]
                        - vectors[b]
                    )

                    repeatability_rows.append({
                        "condition":
                            condition,

                        "mode":
                            mode,

                        "layer":
                            layer_name,

                        "pair":
                            f"{a}-{b}",

                        "joint_count":
                            len(diff),

                        "rmse_deg":
                            float(
                                np.degrees(
                                    safe_rms(
                                        diff
                                    )
                                )
                            ),

                        "mean_abs_deg":
                            float(
                                np.degrees(
                                    np.nanmean(
                                        np.abs(
                                            diff
                                        )
                                    )
                                )
                            ),

                        "max_abs_deg":
                            float(
                                np.degrees(
                                    np.nanmax(
                                        np.abs(
                                            diff
                                        )
                                    )
                                )
                            ),
                    })

    # ============================================================
    # LPF suppression for static trials
    # ============================================================

    filter_rows = []

    for condition in CONDITIONS:
        for round_name in ROUNDS:

            rows02 = loaded[
                (
                    condition,
                    round_name,
                    "alpha02",
                )
            ]

            rows10 = loaded[
                (
                    condition,
                    round_name,
                    "alpha10",
                )
            ]

            q02, v02, names = (
                get_layer_array(
                    rows02,
                    "q_robot_output",
                )
            )

            q10, v10, _ = (
                get_layer_array(
                    rows10,
                    "q_robot_output",
                )
            )

            t = get_relative_time(
                rows02
            )

            mask02 = (
                v02
                & (
                    t
                    >= args.trim_start_s
                )
            )

            mask10 = (
                v10
                & (
                    t
                    >= args.trim_start_s
                )
            )

            dq02, _ = contiguous_delta(
                q02,
                mask02,
                t,
            )

            dq10, _ = contiguous_delta(
                q10,
                mask10,
                t,
            )

            n = min(
                len(dq02),
                len(dq10),
            )

            if n == 0:
                continue

            dq02 = dq02[:n]
            dq10 = dq10[:n]

            for j, name in enumerate(names):

                raw_rms = safe_rms(
                    dq10[:, j]
                )

                filtered_rms = safe_rms(
                    dq02[:, j]
                )

                if (
                    np.isfinite(raw_rms)
                    and raw_rms > 1e-12
                ):
                    reduction = (
                        1.0
                        - filtered_rms
                        / raw_rms
                    ) * 100.0
                else:
                    reduction = np.nan

                filter_rows.append({
                    "condition":
                        condition,
                    "round":
                        round_name,
                    "joint_index":
                        j,
                    "joint_name":
                        name,
                    "joint_short":
                        short_joint_name(
                            name
                        ),
                    "alpha10_delta_rms_deg":
                        float(
                            np.degrees(
                                raw_rms
                            )
                        ),
                    "alpha02_delta_rms_deg":
                        float(
                            np.degrees(
                                filtered_rms
                            )
                        ),
                    "reduction_percent":
                        float(
                            reduction
                        ),
                })

    # ============================================================
    # Save tables
    # ============================================================

    write_csv(
        output_dir
        / "gate3b_static_per_run.csv",
        per_run_rows,
    )

    write_csv(
        output_dir
        / "gate3b_static_per_joint.csv",
        per_joint_rows,
    )

    write_csv(
        output_dir
        / "gate3b_static_condition_summary.csv",
        condition_summary,
    )

    write_csv(
        output_dir
        / "gate3b_static_pose_contrasts.csv",
        contrast_rows,
    )

    write_csv(
        output_dir
        / "gate3b_static_between_run_repeatability.csv",
        repeatability_rows,
    )

    write_csv(
        output_dir
        / "gate3b_static_filter_suppression.csv",
        filter_rows,
    )

    write_csv(
        output_dir
        / "gate3b_static_validation.csv",
        validation_rows,
    )

    # ============================================================
    # Figures
    # ============================================================

    for mode in MODES:

        make_grouped_bar(
            condition_summary,
            mode,
            "mean_of_run_means_deg",
            "Mean joint position (deg)",
            output_dir
            / f"static_pose_joint_means_{mode}.png",
        )

        make_grouped_bar(
            condition_summary,
            mode,
            "mean_within_run_std_deg",
            "Within-run joint std (deg)",
            output_dir
            / f"static_pose_joint_std_{mode}.png",
        )

        make_grouped_bar(
            condition_summary,
            mode,
            "mean_delta_rms_deg_per_frame",
            "Frame-to-frame RMS (deg/frame)",
            output_dir
            / f"static_frame_delta_rms_{mode}.png",
        )

    # Filter suppression figure
    if filter_rows:

        joint_names = []

        for r in filter_rows:

            joint = r["joint_short"]

            # Wrist is a fixed DoF in the current ORCA retargeting setup.
            # Its frame-to-frame RMS is zero, so LPF reduction is undefined.
            if joint == "Wrist":
                continue

            if joint not in joint_names:
                joint_names.append(
                    joint
                )

        values = []

        for joint in joint_names:

            x = [
                r["reduction_percent"]
                for r in filter_rows
                if r["joint_short"] == joint
            ]

            values.append(
                np.nanmean(x)
            )

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.bar(
            np.arange(
                len(joint_names)
            ),
            values,
        )

        ax.set_xticks(
            np.arange(
                len(joint_names)
            )
        )

        ax.set_xticklabels(
            joint_names,
            rotation=60,
            ha="right",
        )

        ax.set_ylabel(
            "Frame-delta RMS reduction (%)"
        )

        ax.set_title(
            "Gate 3B Static — LPF alpha=0.2 suppression"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / "static_filter_suppression.png",
            dpi=180,
        )

        plt.close(fig)

    # ============================================================
    # JSON summary
    # ============================================================

    max_replay_error = np.nanmax(
        [
            r["replay_max_abs_rad"]
            for r in validation_rows
        ]
    )

    summary = {
        "gate": "Gate 3B-3 Static Mapping Validation",

        "input_directory":
            str(input_dir),

        "trim_start_s":
            args.trim_start_s,

        "expected_static_files":
            18,

        "analyzed_static_files":
            len(loaded),

        "conditions":
            CONDITIONS,

        "rounds":
            ROUNDS,

        "modes":
            MODES,

        "max_replay_validation_error_rad":
            float(
                max_replay_error
            ),

        "max_replay_validation_error_deg":
            float(
                np.degrees(
                    max_replay_error
                )
            ),

        "notes": [
            (
                "alpha02 replay validation compares "
                "q_robot_output with Gate3A logged q."
            ),
            (
                "alpha10 replay validation compares "
                "q_robot_output with q_robot_prefilter."
            ),
            (
                "steady-state statistics exclude the "
                f"first {args.trim_start_s:.3f} s."
            ),
        ],
    }

    with (
        output_dir
        / "gate3b_static_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("Gate 3B-3 Static Mapping Analysis")
    print("=" * 72)
    print("Static replay files :", len(loaded))
    print(
        "Max replay error    :",
        max_replay_error,
        "rad",
    )
    print("Output directory    :", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()