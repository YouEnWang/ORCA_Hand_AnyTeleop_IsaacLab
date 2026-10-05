#!/usr/bin/env python3

"""
Gate 3B Dynamic Analysis

Analyzes:

    slow_open_close R1-R3
    occlusion       R1-R3

for:

    alpha02 = canonical LPF (alpha=0.2)
    alpha10 = LPF bypass     (alpha=1.0)

Main goals
----------
Slow Open-Close:
    - command variation
    - range preservation
    - LPF smoothing
    - LPF lag
    - amplitude ratio

Occlusion:
    - exact invalid/dropout episodes
    - command behavior before dropout
    - first second after reacquisition
    - post-reacquisition stabilization
    - alpha02 vs alpha10 behavior

No invalid frames are interpolated.

Outputs
-------
results/gate3b/dynamic/

    gate3b_dynamic_per_run.csv
    slow_open_close_per_joint.csv
    slow_open_close_filter_effect.csv
    occlusion_dropout_episodes.csv
    occlusion_window_metrics.csv
    occlusion_post_reacquisition.csv
    gate3b_dynamic_summary.json

    slow_open_close_delta_rms.png
    slow_open_close_lag.png
    occlusion_validity_timeline.png
    occlusion_post_reacquisition_delta.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROUNDS = ["r1", "r2", "r3"]

MODES = {
    "alpha02": 0.2,
    "alpha10": 1.0,
}


def short_joint_name(name: str) -> str:

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

    return mapping.get(
        name,
        name,
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Gate 3B dynamic motion analysis"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "data/gate3b/replay"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/gate3b/dynamic"
        ),
    )

    parser.add_argument(
        "--max-lag-s",
        type=float,
        default=1.0,
        help=(
            "Maximum lag searched between alpha10 "
            "and alpha02 trajectories."
        ),
    )

    parser.add_argument(
        "--post-reacq-s",
        type=float,
        default=1.0,
        help=(
            "Post-reacquisition analysis duration."
        ),
    )

    parser.add_argument(
        "--stabilization-window-s",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--stabilization-sigma",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def load_jsonl(path):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def relative_time(rows):

    t = np.asarray(
        [
            r.get(
                "capture_monotonic_s"
            )
            for r in rows
        ],
        dtype=float,
    )

    finite = np.isfinite(t)

    if np.any(finite):
        t0 = t[
            np.where(
                finite
            )[0][0]
        ]

        return t - t0

    return (
        np.arange(len(rows))
        / 30.0
    )


def get_q(rows, key="q_robot_output"):

    names = None

    for r in rows:
        if r.get(
            "robot_joint_names"
        ):
            names = list(
                r[
                    "robot_joint_names"
                ]
            )
            break

    if names is None:
        raise RuntimeError(
            "robot_joint_names missing"
        )

    q = np.full(
        (
            len(rows),
            len(names),
        ),
        np.nan,
    )

    valid = np.zeros(
        len(rows),
        dtype=bool,
    )

    for i, r in enumerate(rows):

        if not r.get(
            "tracking_valid",
            False,
        ):
            continue

        x = r.get(key)

        if x is None:
            continue

        x = np.asarray(
            x,
            dtype=float,
        )

        if (
            x.shape
            != (
                len(names),
            )
        ):
            continue

        if not np.all(
            np.isfinite(x)
        ):
            continue

        q[i] = x
        valid[i] = True

    return q, valid, names


def median_dt(t):

    dt = np.diff(t)

    dt = dt[
        np.isfinite(dt)
        & (dt > 0)
    ]

    if not len(dt):
        return 1.0 / 30.0

    return float(
        np.median(dt)
    )


def consecutive_derivatives(
    q,
    valid,
    t,
):
    """
    Return dq/frame, velocity, acceleration.

    Values spanning invalid-frame gaps are discarded.
    """

    n, d = q.shape

    dq = np.full(
        (n, d),
        np.nan,
    )

    vel = np.full(
        (n, d),
        np.nan,
    )

    acc = np.full(
        (n, d),
        np.nan,
    )

    med_dt = median_dt(t)
    max_dt = 2.5 * med_dt

    for i in range(
        1,
        n,
    ):

        if not (
            valid[i]
            and valid[i - 1]
        ):
            continue

        dt = (
            t[i]
            - t[i - 1]
        )

        if (
            not np.isfinite(dt)
            or dt <= 0
            or dt > max_dt
        ):
            continue

        dq[i] = (
            q[i]
            - q[i - 1]
        )

        vel[i] = (
            dq[i]
            / dt
        )

    for i in range(
        2,
        n,
    ):

        if (
            not np.all(
                np.isfinite(
                    vel[i]
                )
            )
            or not np.all(
                np.isfinite(
                    vel[i - 1]
                )
            )
        ):
            continue

        dt = (
            t[i]
            - t[i - 1]
        )

        if (
            dt <= 0
            or dt > max_dt
        ):
            continue

        acc[i] = (
            vel[i]
            - vel[i - 1]
        ) / dt

    return (
        dq,
        vel,
        acc,
    )


def rms(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if not len(x):
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                x ** 2
            )
        )
    )


def estimate_lag_frames(
    reference,
    filtered,
    max_lag_frames,
):
    """
    Positive lag means filtered signal trails reference.

    Uses first difference to reduce bias from long static holds.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    filtered = np.asarray(
        filtered,
        dtype=float,
    )

    mask = (
        np.isfinite(reference)
        & np.isfinite(filtered)
    )

    reference = reference[mask]
    filtered = filtered[mask]

    if len(reference) < 20:
        return np.nan, np.nan

    dr = np.diff(reference)
    df = np.diff(filtered)

    if (
        np.std(dr) < 1e-8
        or np.std(df) < 1e-8
    ):
        return np.nan, np.nan

    best_lag = None
    best_corr = -np.inf

    for lag in range(
        -max_lag_frames,
        max_lag_frames + 1,
    ):

        if lag > 0:
            a = dr[:-lag]
            b = df[lag:]

        elif lag < 0:
            a = dr[-lag:]
            b = df[:lag]

        else:
            a = dr
            b = df

        if len(a) < 10:
            continue

        if (
            np.std(a) < 1e-8
            or np.std(b) < 1e-8
        ):
            continue

        corr = np.corrcoef(
            a,
            b,
        )[0, 1]

        if (
            np.isfinite(corr)
            and corr > best_corr
        ):
            best_corr = corr
            best_lag = lag

    if best_lag is None:
        return np.nan, np.nan

    return (
        int(best_lag),
        float(best_corr),
    )


def find_invalid_episodes(
    valid,
    t,
):
    """
    Return contiguous tracking-invalid episodes.
    """

    episodes = []

    n = len(valid)

    i = 0

    while i < n:

        if valid[i]:
            i += 1
            continue

        start = i

        while (
            i + 1 < n
            and not valid[i + 1]
        ):
            i += 1

        end = i

        first_valid_after = (
            end + 1
            if end + 1 < n
            else None
        )

        dt = median_dt(t)

        duration = (
            t[end] - t[start] + dt
        )

        episodes.append({
            "start_index":
                start,

            "end_index":
                end,

            "frames":
                end - start + 1,

            "start_s":
                float(t[start]),

            "end_s":
                float(t[end]),

            "duration_s":
                float(duration),

            "first_valid_after_index":
                first_valid_after,

            "first_valid_after_s":
                (
                    float(
                        t[
                            first_valid_after
                        ]
                    )
                    if first_valid_after
                    is not None
                    else np.nan
                ),
        })

        i += 1

    return episodes


def window_q_metrics(
    q,
    valid,
    t,
    start,
    end,
):
    mask = (
        valid
        & (t >= start)
        & (t < end)
    )

    dq, vel, acc = (
        consecutive_derivatives(
            q,
            mask,
            t,
        )
    )

    return {
        "valid_frames":
            int(
                np.sum(mask)
            ),

        "delta_rms_deg_per_frame":
            float(
                np.degrees(
                    rms(dq)
                )
            ),

        "delta_p95_deg_per_frame":
            float(
                np.degrees(
                    np.nanpercentile(
                        np.abs(
                            dq[
                                np.isfinite(
                                    dq
                                )
                            ]
                        ),
                        95,
                    )
                )
            )
            if np.any(
                np.isfinite(dq)
            )
            else np.nan,

        "velocity_rms_deg_s":
            float(
                np.degrees(
                    rms(vel)
                )
            ),

        "acceleration_rms_deg_s2":
            float(
                np.degrees(
                    rms(acc)
                )
            ),
    }


def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    args = parse_args()

    input_dir = (
        args.input_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Slow Open-Close
    # ============================================================

    slow_data = {}

    dynamic_run_rows = []
    slow_joint_rows = []
    slow_filter_rows = []

    for round_name in ROUNDS:
        for mode, alpha in MODES.items():

            path = (
                input_dir
                / (
                    f"slow_open_close_"
                    f"{round_name}_"
                    f"{mode}.jsonl"
                )
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            rows = load_jsonl(path)

            slow_data[
                (
                    round_name,
                    mode,
                )
            ] = rows

            q, valid, names = get_q(
                rows
            )

            t = relative_time(
                rows
            )

            dq, vel, acc = (
                consecutive_derivatives(
                    q,
                    valid,
                    t,
                )
            )

            dynamic_run_rows.append({
                "motion":
                    "slow_open_close",

                "round":
                    round_name,

                "mode":
                    mode,

                "alpha":
                    alpha,

                "frames":
                    len(rows),

                "valid_frames":
                    int(valid.sum()),

                "delta_rms_deg_per_frame":
                    float(
                        np.degrees(
                            rms(dq)
                        )
                    ),

                "velocity_rms_deg_s":
                    float(
                        np.degrees(
                            rms(vel)
                        )
                    ),

                "acceleration_rms_deg_s2":
                    float(
                        np.degrees(
                            rms(acc)
                        )
                    ),
            })

            for j, name in enumerate(
                names
            ):

                x = q[:, j]

                finite = (
                    valid
                    & np.isfinite(x)
                )

                slow_joint_rows.append({
                    "round":
                        round_name,

                    "mode":
                        mode,

                    "alpha":
                        alpha,

                    "joint_index":
                        j,

                    "joint_name":
                        name,

                    "joint_short":
                        short_joint_name(
                            name
                        ),

                    "range_deg":
                        float(
                            np.degrees(
                                np.nanmax(
                                    x[finite]
                                )
                                -
                                np.nanmin(
                                    x[finite]
                                )
                            )
                        ),

                    "delta_rms_deg_per_frame":
                        float(
                            np.degrees(
                                rms(
                                    dq[:, j]
                                )
                            )
                        ),

                    "velocity_rms_deg_s":
                        float(
                            np.degrees(
                                rms(
                                    vel[:, j]
                                )
                            )
                        ),
                })

    # ------------------------------------------------------------
    # Slow motion alpha02 vs alpha10
    # ------------------------------------------------------------

    for round_name in ROUNDS:

        rows02 = slow_data[
            (
                round_name,
                "alpha02",
            )
        ]

        rows10 = slow_data[
            (
                round_name,
                "alpha10",
            )
        ]

        q02, valid02, names = get_q(
            rows02
        )

        q10, valid10, _ = get_q(
            rows10
        )

        t = relative_time(
            rows10
        )

        dt = median_dt(t)

        max_lag_frames = int(
            round(
                args.max_lag_s
                / dt
            )
        )

        dq02, _, _ = (
            consecutive_derivatives(
                q02,
                valid02,
                t,
            )
        )

        dq10, _, _ = (
            consecutive_derivatives(
                q10,
                valid10,
                t,
            )
        )

        for j, name in enumerate(
            names
        ):

            raw_rms = rms(
                dq10[:, j]
            )

            filt_rms = rms(
                dq02[:, j]
            )

            if (
                np.isfinite(raw_rms)
                and raw_rms > 1e-12
            ):
                reduction = (
                    1
                    - filt_rms
                    / raw_rms
                ) * 100.0
            else:
                reduction = np.nan

            mask10 = (
                valid10
                & np.isfinite(
                    q10[:, j]
                )
            )

            mask02 = (
                valid02
                & np.isfinite(
                    q02[:, j]
                )
            )

            range10 = (
                np.nanmax(
                    q10[
                        mask10,
                        j,
                    ]
                )
                -
                np.nanmin(
                    q10[
                        mask10,
                        j,
                    ]
                )
            )

            range02 = (
                np.nanmax(
                    q02[
                        mask02,
                        j,
                    ]
                )
                -
                np.nanmin(
                    q02[
                        mask02,
                        j,
                    ]
                )
            )

            amplitude_ratio = (
                range02
                / range10
                if range10 > 1e-12
                else np.nan
            )

            lag_frames, corr = (
                estimate_lag_frames(
                    q10[:, j],
                    q02[:, j],
                    max_lag_frames,
                )
            )

            lag_ms = (
                lag_frames
                * dt
                * 1000.0
                if np.isfinite(
                    lag_frames
                )
                else np.nan
            )

            slow_filter_rows.append({
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
                            filt_rms
                        )
                    ),

                "delta_rms_reduction_percent":
                    float(
                        reduction
                    ),

                "alpha10_range_deg":
                    float(
                        np.degrees(
                            range10
                        )
                    ),

                "alpha02_range_deg":
                    float(
                        np.degrees(
                            range02
                        )
                    ),

                "amplitude_ratio":
                    float(
                        amplitude_ratio
                    ),

                "lag_frames":
                    float(
                        lag_frames
                    ),

                "lag_ms":
                    float(
                        lag_ms
                    ),

                "lag_correlation":
                    float(
                        corr
                    ),
            })

    # ============================================================
    # Occlusion / Recovery
    # ============================================================

    occlusion_data = {}

    dropout_rows = []
    window_rows = []
    reacq_rows = []

    # These windows follow the observed Gate 3A analysis.
    # They are intentionally configurable in code rather than
    # interpreted as detector ground truth.
    windows = {
        "clean_baseline":
            (2.0, 5.0),

        "partial_observed":
            (5.0, 9.0),

        "recovery1":
            (10.0, 12.0),

        "self_occlusion_observed":
            (12.0, 19.0),

        "recovery2":
            (19.0, 21.0),

        "final_recovery":
            (27.0, 30.0),
    }

    for round_name in ROUNDS:
        for mode, alpha in MODES.items():

            path = (
                input_dir
                / (
                    f"occlusion_"
                    f"{round_name}_"
                    f"{mode}.jsonl"
                )
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            rows = load_jsonl(path)

            occlusion_data[
                (
                    round_name,
                    mode,
                )
            ] = rows

            q, valid, names = get_q(
                rows
            )

            t = relative_time(
                rows
            )

            dq, vel, acc = (
                consecutive_derivatives(
                    q,
                    valid,
                    t,
                )
            )

            dynamic_run_rows.append({
                "motion":
                    "occlusion",

                "round":
                    round_name,

                "mode":
                    mode,

                "alpha":
                    alpha,

                "frames":
                    len(rows),

                "valid_frames":
                    int(valid.sum()),

                "delta_rms_deg_per_frame":
                    float(
                        np.degrees(
                            rms(dq)
                        )
                    ),

                "velocity_rms_deg_s":
                    float(
                        np.degrees(
                            rms(vel)
                        )
                    ),

                "acceleration_rms_deg_s2":
                    float(
                        np.degrees(
                            rms(acc)
                        )
                    ),
            })

            # --------------------------------------------
            # Window metrics
            # --------------------------------------------

            for label, (
                start,
                end,
            ) in windows.items():

                m = window_q_metrics(
                    q,
                    valid,
                    t,
                    start,
                    end,
                )

                row = {
                    "round":
                        round_name,

                    "mode":
                        mode,

                    "alpha":
                        alpha,

                    "window":
                        label,

                    "start_s":
                        start,

                    "end_s":
                        end,
                }

                row.update(m)

                window_rows.append(
                    row
                )

            # --------------------------------------------
            # Invalid episodes
            # Same tracking mask for alpha02/alpha10,
            # therefore record only once.
            # --------------------------------------------

            if mode == "alpha02":

                episodes = (
                    find_invalid_episodes(
                        valid,
                        t,
                    )
                )

                for episode_id, ep in enumerate(
                    episodes,
                    start=1,
                ):

                    row = {
                        "round":
                            round_name,

                        "episode":
                            episode_id,
                    }

                    row.update(ep)

                    dropout_rows.append(
                        row
                    )

            # --------------------------------------------
            # Post-reacquisition command transient
            # --------------------------------------------

            episodes = find_invalid_episodes(
                valid,
                t,
            )

            for episode_id, ep in enumerate(
                episodes,
                start=1,
            ):

                first_valid = (
                    ep[
                        "first_valid_after_index"
                    ]
                )

                if first_valid is None:
                    continue

                start_s = t[first_valid]

                end_s = (
                    start_s
                    + args.post_reacq_s
                )

                m = window_q_metrics(
                    q,
                    valid,
                    t,
                    start_s,
                    end_s,
                )

                # Jump from last valid frame before dropout
                # to first valid frame after dropout.
                last_valid_before = (
                    ep["start_index"] - 1
                )

                if (
                    last_valid_before >= 0
                    and valid[
                        last_valid_before
                    ]
                    and valid[
                        first_valid
                    ]
                ):

                    jump = (
                        q[first_valid]
                        - q[
                            last_valid_before
                        ]
                    )

                    mean_abs_jump_deg = (
                        np.degrees(
                            np.mean(
                                np.abs(
                                    jump
                                )
                            )
                        )
                    )

                    max_abs_jump_deg = (
                        np.degrees(
                            np.max(
                                np.abs(
                                    jump
                                )
                            )
                        )
                    )

                else:
                    mean_abs_jump_deg = np.nan
                    max_abs_jump_deg = np.nan

                reacq_rows.append({
                    "round":
                        round_name,

                    "mode":
                        mode,

                    "alpha":
                        alpha,

                    "episode":
                        episode_id,

                    "first_valid_after_s":
                        float(
                            start_s
                        ),

                    "analysis_duration_s":
                        args.post_reacq_s,

                    "mean_abs_reentry_jump_deg":
                        float(
                            mean_abs_jump_deg
                        ),

                    "max_abs_reentry_jump_deg":
                        float(
                            max_abs_jump_deg
                        ),

                    **m,
                })

    # ============================================================
    # Save CSV
    # ============================================================

    write_csv(
        output_dir
        / "gate3b_dynamic_per_run.csv",
        dynamic_run_rows,
    )

    write_csv(
        output_dir
        / "slow_open_close_per_joint.csv",
        slow_joint_rows,
    )

    write_csv(
        output_dir
        / "slow_open_close_filter_effect.csv",
        slow_filter_rows,
    )

    write_csv(
        output_dir
        / "occlusion_dropout_episodes.csv",
        dropout_rows,
    )

    write_csv(
        output_dir
        / "occlusion_window_metrics.csv",
        window_rows,
    )

    write_csv(
        output_dir
        / "occlusion_post_reacquisition.csv",
        reacq_rows,
    )

    # ============================================================
    # Figures — slow open-close delta RMS
    # ============================================================

    joint_names = []

    for r in slow_filter_rows:
        
        joint = r["joint_short"]
        
        # Wrist is a fixed DoF in the current ORCA retargeting setup.
        # Its frame-to-frame RMS is zero, so LPF reduction and lag
        # are not meaningful.
        if joint == "Wrist":
            continue

        if joint not in joint_names:
            joint_names.append(
                joint
            )

    if joint_names:

        raw_vals = []
        filt_vals = []
        lag_vals = []

        for joint in joint_names:

            jr = [
                r
                for r in slow_filter_rows
                if r["joint_short"]
                == joint
            ]

            raw_vals.append(
                np.nanmean(
                    [
                        r[
                            "alpha10_delta_rms_deg"
                        ]
                        for r in jr
                    ]
                )
            )

            filt_vals.append(
                np.nanmean(
                    [
                        r[
                            "alpha02_delta_rms_deg"
                        ]
                        for r in jr
                    ]
                )
            )

            lag_vals.append(
                np.nanmean(
                    [
                        r["lag_ms"]
                        for r in jr
                    ]
                )
            )

        x = np.arange(
            len(joint_names)
        )

        width = 0.38

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.bar(
            x - width / 2,
            raw_vals,
            width,
            label="alpha=1.0",
        )

        ax.bar(
            x + width / 2,
            filt_vals,
            width,
            label="alpha=0.2",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            joint_names,
            rotation=60,
            ha="right",
        )

        ax.set_ylabel(
            "Frame-to-frame RMS (deg/frame)"
        )

        ax.set_title(
            "Slow Open-Close — Command Variation"
        )

        ax.legend()

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / "slow_open_close_delta_rms.png",
            dpi=180,
        )

        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(15, 6)
        )

        ax.bar(
            x,
            lag_vals,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            joint_names,
            rotation=60,
            ha="right",
        )

        ax.set_ylabel(
            "Estimated LPF lag (ms)"
        )

        ax.set_title(
            "Slow Open-Close — alpha=0.2 Estimated Lag"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / "slow_open_close_lag.png",
            dpi=180,
        )

        plt.close(fig)

    # ============================================================
    # Occlusion validity timeline
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    for k, round_name in enumerate(
        ROUNDS
    ):

        rows = occlusion_data[
            (
                round_name,
                "alpha02",
            )
        ]

        t = relative_time(
            rows
        )

        _, valid, _ = get_q(
            rows
        )

        y = (
            valid.astype(float)
            + k * 1.3
        )

        ax.step(
            t,
            y,
            where="post",
            label=round_name,
        )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Tracking-valid timeline"
    )

    ax.set_title(
        "Occlusion — Tracking Availability"
    )

    ax.legend()

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "occlusion_validity_timeline.png",
        dpi=180,
    )

    plt.close(fig)

    # ============================================================
    # Post-reacquisition delta
    # ============================================================

    if reacq_rows:

        labels = []
        alpha10 = []
        alpha02 = []

        for round_name in ROUNDS:

            labels.append(
                round_name
            )

            r10 = [
                r
                for r in reacq_rows
                if r["round"]
                == round_name
                and r["mode"]
                == "alpha10"
            ]

            r02 = [
                r
                for r in reacq_rows
                if r["round"]
                == round_name
                and r["mode"]
                == "alpha02"
            ]

            alpha10.append(
                np.nanmean(
                    [
                        r[
                            "delta_rms_deg_per_frame"
                        ]
                        for r in r10
                    ]
                )
            )

            alpha02.append(
                np.nanmean(
                    [
                        r[
                            "delta_rms_deg_per_frame"
                        ]
                        for r in r02
                    ]
                )
            )

        x = np.arange(
            len(labels)
        )

        width = 0.35

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            x - width / 2,
            alpha10,
            width,
            label="alpha=1.0",
        )

        ax.bar(
            x + width / 2,
            alpha02,
            width,
            label="alpha=0.2",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels
        )

        ax.set_ylabel(
            "Post-reacquisition RMS (deg/frame)"
        )

        ax.set_title(
            "Occlusion — First 1 s After Reacquisition"
        )

        ax.legend()

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / "occlusion_post_reacquisition_delta.png",
            dpi=180,
        )

        plt.close(fig)

    # ============================================================
    # JSON summary
    # ============================================================

    summary = {
        "gate":
            "Gate 3B Dynamic Analysis",

        "input_directory":
            str(input_dir),

        "slow_open_close_files":
            6,

        "occlusion_files":
            6,

        "post_reacquisition_window_s":
            args.post_reacq_s,

        "max_lag_search_s":
            args.max_lag_s,

        "occlusion_windows":
            windows,

        "important_interpretation_notes": [
            (
                "Positive lag means alpha=0.2 "
                "trails alpha=1.0."
            ),
            (
                "Lag is estimated from first differences "
                "rather than absolute joint position."
            ),
            (
                "Dropout episodes are derived directly from "
                "tracking_valid=False."
            ),
            (
                "The partial/self-occlusion windows are "
                "approximate observed analysis windows, "
                "not detector ground-truth event timestamps."
            ),
            (
                "Re-entry jump includes physical hand "
                "repositioning and must not be interpreted "
                "as pure estimator error."
            ),
        ],
    }

    with (
        output_dir
        / "gate3b_dynamic_summary.json"
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
    print("Gate 3B Dynamic Analysis Complete")
    print("=" * 72)
    print(
        "Slow Open-Close files : 6"
    )
    print(
        "Occlusion files       : 6"
    )
    print(
        "Output directory      :",
        output_dir,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()