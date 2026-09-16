#!/usr/bin/env python3

"""
Analyze Gate 2B ORCA joint-response experiments.

Supported experiments:
    static
    step
    sine

Outputs
-------
static:
    static_joint_metrics.csv
    static_summary.json

step:
    step_event_metrics.csv
    step_summary.json

sine:
    sine_event_metrics.csv
    sine_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_npz(
    path: Path,
) -> dict[str, np.ndarray]:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_json(
    path: Path,
    obj: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def convert(value):
        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            (np.integer,),
        ):
            return int(value)

        if isinstance(
            value,
            (np.floating,),
        ):
            return float(value)

        return value

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=convert,
        )


def hf_energy_ratio(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float = 6.0,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    centered = (
        values
        - np.mean(values)
    )

    spectrum = np.fft.rfft(
        centered
    )

    power = (
        np.abs(spectrum)
        ** 2
    )

    frequencies = np.fft.rfftfreq(
        len(values),
        d=1.0 / sample_rate_hz,
    )

    total = float(
        np.sum(power)
    )

    if total <= 1e-20:
        return 0.0

    return float(
        np.sum(
            power[
                frequencies
                >= cutoff_hz
            ]
        )
        / total
    )


def derivative_rms(
    values: np.ndarray,
    dt: float,
    order: int,
) -> float:
    result = np.asarray(
        values,
        dtype=np.float64,
    )

    for _ in range(order):
        result = np.gradient(
            result,
            dt,
        )

    return float(
        np.sqrt(
            np.mean(
                result ** 2
            )
        )
    )


def first_crossing_time(
    time_values: np.ndarray,
    normalized_response: np.ndarray,
    threshold: float,
) -> float:
    idx = np.flatnonzero(
        normalized_response
        >= threshold
    )

    if idx.size == 0:
        return float("nan")

    return float(
        time_values[
            idx[0]
        ]
    )


def settling_time(
    time_values: np.ndarray,
    actual: np.ndarray,
    target: float,
    amplitude: float,
    tolerance_fraction: float = 0.02,
) -> float:
    band = (
        tolerance_fraction
        * abs(amplitude)
    )

    if band <= 1e-12:
        return float("nan")

    inside = (
        np.abs(
            actual - target
        )
        <= band
    )

    for idx in range(
        len(inside)
    ):
        if np.all(
            inside[idx:]
        ):
            return float(
                time_values[idx]
                - time_values[0]
            )

    return float("nan")


def harmonic_fit(
    time_values: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Fit:
        y = a*sin(wt) + b*cos(wt) + c

    Returns:
        amplitude
        phase_rad
        offset
    """

    omega = (
        2.0
        * np.pi
        * frequency_hz
    )

    matrix = np.column_stack(
        [
            np.sin(
                omega
                * time_values
            ),
            np.cos(
                omega
                * time_values
            ),
            np.ones_like(
                time_values
            ),
        ]
    )

    coeff, *_ = np.linalg.lstsq(
        matrix,
        values,
        rcond=None,
    )

    a, b, offset = coeff

    amplitude = float(
        np.sqrt(
            a * a
            + b * b
        )
    )

    phase = float(
        np.arctan2(
            b,
            a,
        )
    )

    return (
        amplitude,
        phase,
        float(offset),
    )


def analyze_static(
    data: dict,
    output_dir: Path,
    sample_rate_hz: float,
) -> None:

    time_values = data["time"]
    q_command = data["q_command"]
    q_actual = data["q_actual"]
    joint_names = data["joint_names"]

    dt = float(
        np.median(
            np.diff(
                time_values
            )
        )
    )

    rows = []

    for joint_index, name in enumerate(
        joint_names
    ):
        command = q_command[
            :,
            joint_index,
        ]

        actual = q_actual[
            :,
            joint_index,
        ]

        error = (
            actual
            - command
        )

        rows.append(
            {
                "joint_index":
                    joint_index,

                "joint_name":
                    str(name),

                "command_mean_rad":
                    float(
                        np.mean(
                            command
                        )
                    ),

                "actual_mean_rad":
                    float(
                        np.mean(
                            actual
                        )
                    ),

                "bias_rad":
                    float(
                        np.mean(
                            error
                        )
                    ),

                "rmse_rad":
                    float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),

                "actual_std_rad":
                    float(
                        np.std(
                            actual
                        )
                    ),

                "actual_peak_to_peak_rad":
                    float(
                        np.ptp(
                            actual
                        )
                    ),

                "hf_energy_ratio":
                    hf_energy_ratio(
                        actual,
                        sample_rate_hz,
                    ),

                "velocity_rms_rad_s":
                    derivative_rms(
                        actual,
                        dt,
                        1,
                    ),

                "acceleration_rms_rad_s2":
                    derivative_rms(
                        actual,
                        dt,
                        2,
                    ),

                "jerk_rms_rad_s3":
                    derivative_rms(
                        actual,
                        dt,
                        3,
                    ),
            }
        )

    write_csv(
        output_dir
        / "static_joint_metrics.csv",
        rows,
    )

    summary = {
        "experiment":
            "static",

        "sample_rate_hz":
            sample_rate_hz,

        "duration_s":
            float(
                time_values[-1]
                - time_values[0]
            ),

        "mean_actual_std_rad":
            float(
                np.mean(
                    [
                        row[
                            "actual_std_rad"
                        ]
                        for row in rows
                    ]
                )
            ),

        "max_actual_std_rad":
            float(
                np.max(
                    [
                        row[
                            "actual_std_rad"
                        ]
                        for row in rows
                    ]
                )
            ),

        "mean_rmse_rad":
            float(
                np.mean(
                    [
                        row[
                            "rmse_rad"
                        ]
                        for row in rows
                    ]
                )
            ),

        "max_rmse_rad":
            float(
                np.max(
                    [
                        row[
                            "rmse_rad"
                        ]
                        for row in rows
                    ]
                )
            ),
    }

    write_json(
        output_dir
        / "static_summary.json",
        summary,
    )

    print(
        "[PASS] Static analysis complete."
    )

    print(
        f"Mean actual std : "
        f"{summary['mean_actual_std_rad']:.8f} rad"
    )

    print(
        f"Max actual std  : "
        f"{summary['max_actual_std_rad']:.8f} rad"
    )


def analyze_step(
    data: dict,
    output_dir: Path,
    sample_rate_hz: float,
) -> None:

    time_values = data["time"]
    q_desired = data["q_desired"]
    q_command = data["q_command"]
    q_actual = data["q_actual"]

    phase = data["phase"]
    event_ids = data["event_id"]
    active_joint = data[
        "active_joint_index"
    ]

    joint_names = data[
        "joint_names"
    ]

    rows = []

    unique_events = np.unique(
        event_ids[
            event_ids >= 0
        ]
    )

    for event_id in unique_events:

        event_mask = (
            event_ids
            == event_id
        )

        step_mask = (
            event_mask
            & (phase == "step")
        )

        pre_mask = (
            event_mask
            & (phase == "pre")
        )

        if not np.any(
            step_mask
        ):
            continue

        joint_index = int(
            active_joint[
                np.flatnonzero(
                    step_mask
                )[0]
            ]
        )

        name = str(
            joint_names[
                joint_index
            ]
        )

        t_step = time_values[
            step_mask
        ]

        desired = q_desired[
            step_mask,
            joint_index,
        ]

        command = q_command[
            step_mask,
            joint_index,
        ]

        actual = q_actual[
            step_mask,
            joint_index,
        ]

        pre_command = q_command[
            pre_mask,
            joint_index,
        ]

        pre_actual = q_actual[
            pre_mask,
            joint_index,
        ]

        pre_tail = max(
            1,
            int(
                round(
                    0.2
                    * len(
                        pre_actual
                    )
                )
            ),
        )

        baseline_command = float(
            np.mean(
                pre_command[
                    -pre_tail:
                ]
            )
        )

        baseline_actual = float(
            np.mean(
                pre_actual[
                    -pre_tail:
                ]
            )
        )

        step_tail = max(
            1,
            int(
                round(
                    0.2
                    * len(
                        command
                    )
                )
            ),
        )

        target_command = float(
            np.mean(
                command[
                    -step_tail:
                ]
            )
        )

        amplitude = (
            target_command
            - baseline_command
        )

        direction = (
            1.0
            if amplitude >= 0.0
            else -1.0
        )

        abs_amplitude = abs(
            amplitude
        )

        if abs_amplitude <= 1e-12:
            continue

        normalized_command = (
            direction
            * (
                command
                - baseline_command
            )
            / abs_amplitude
        )

        normalized_actual = (
            direction
            * (
                actual
                - baseline_actual
            )
            / abs_amplitude
        )

        relative_time = (
            t_step
            - t_step[0]
        )

        cmd_05 = first_crossing_time(
            relative_time,
            normalized_command,
            0.05,
        )

        cmd_10 = first_crossing_time(
            relative_time,
            normalized_command,
            0.10,
        )

        cmd_90 = first_crossing_time(
            relative_time,
            normalized_command,
            0.90,
        )

        act_05 = first_crossing_time(
            relative_time,
            normalized_actual,
            0.05,
        )

        act_10 = first_crossing_time(
            relative_time,
            normalized_actual,
            0.10,
        )

        act_90 = first_crossing_time(
            relative_time,
            normalized_actual,
            0.90,
        )

        command_rise_time = (
            cmd_90
            - cmd_10
        )

        actual_rise_time = (
            act_90
            - act_10
        )

        response_delay = (
            act_05
            - cmd_05
        )

        overshoot_percent = (
            np.max(
                normalized_actual
            )
            - 1.0
        ) * 100.0

        actual_settling_time = (
            settling_time(
                relative_time,
                actual,
                target_command,
                amplitude,
                tolerance_fraction=0.02,
            )
        )

        steady_tail = max(
            1,
            int(
                round(
                    0.2
                    * len(
                        actual
                    )
                )
            ),
        )

        steady_actual = float(
            np.mean(
                actual[
                    -steady_tail:
                ]
            )
        )

        steady_error = (
            steady_actual
            - target_command
        )

        tracking_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        actual
                        - command
                    )
                    ** 2
                )
            )
        )

        # ------------------------------------------------------
        # Cross-joint coupling
        # ------------------------------------------------------

        other_indices = [
            idx
            for idx in range(
                q_actual.shape[1]
            )
            if idx != joint_index
        ]

        baseline_other = np.mean(
            q_actual[
                pre_mask,
                :
            ][
                -pre_tail:,
                :
            ],
            axis=0,
        )

        if other_indices:
            other_deviation = np.abs(
                q_actual[
                    step_mask,
                    :
                ][:, other_indices]
                - baseline_other[
                    other_indices
                ]
            )

            cross_coupling_max = float(
                np.max(
                    other_deviation
                )
            )

        else:
            cross_coupling_max = 0.0

        rows.append(
            {
                "event_id":
                    int(event_id),

                "joint_index":
                    joint_index,

                "joint_name":
                    name,

                "target_command_rad":
                    target_command,

                "step_amplitude_rad":
                    amplitude,

                "command_10_90_rise_s":
                    command_rise_time,

                "command_to_actual_delay_5pct_s":
                    response_delay,

                "actual_10_90_rise_s":
                    actual_rise_time,

                "overshoot_percent":
                    float(
                        overshoot_percent
                    ),

                "settling_time_2pct_s":
                    actual_settling_time,

                "steady_state_error_rad":
                    steady_error,

                "command_actual_rmse_rad":
                    tracking_rmse,

                "cross_joint_max_deviation_rad":
                    cross_coupling_max,
            }
        )

    write_csv(
        output_dir
        / "step_event_metrics.csv",
        rows,
    )

    overshoot_values = np.asarray(
        [
            row[
                "overshoot_percent"
            ]
            for row in rows
        ],
        dtype=float,
    )

    rmse_values = np.asarray(
        [
            row[
                "command_actual_rmse_rad"
            ]
            for row in rows
        ],
        dtype=float,
    )

    summary = {
        "experiment":
            "step",

        "event_count":
            len(rows),

        "sample_rate_hz":
            sample_rate_hz,

        "mean_overshoot_percent":
            float(
                np.nanmean(
                    overshoot_values
                )
            ),

        "max_overshoot_percent":
            float(
                np.nanmax(
                    overshoot_values
                )
            ),

        "mean_command_actual_rmse_rad":
            float(
                np.nanmean(
                    rmse_values
                )
            ),

        "max_command_actual_rmse_rad":
            float(
                np.nanmax(
                    rmse_values
                )
            ),
    }

    write_json(
        output_dir
        / "step_summary.json",
        summary,
    )

    print(
        "[PASS] Step analysis complete."
    )

    print(
        f"Events          : {len(rows)}"
    )

    print(
        f"Mean overshoot  : "
        f"{summary['mean_overshoot_percent']:.2f}%"
    )

    print(
        f"Max overshoot   : "
        f"{summary['max_overshoot_percent']:.2f}%"
    )


def analyze_sine(
    data: dict,
    output_dir: Path,
    sample_rate_hz: float,
) -> None:

    time_values = data["time"]
    q_command = data["q_command"]
    q_actual = data["q_actual"]

    phase = data["phase"]
    event_ids = data["event_id"]
    active_joint = data[
        "active_joint_index"
    ]
    frequencies = data[
        "frequency_hz"
    ]

    joint_names = data[
        "joint_names"
    ]

    rows = []

    unique_events = np.unique(
        event_ids[
            event_ids >= 0
        ]
    )

    for event_id in unique_events:

        event_mask = (
            event_ids
            == event_id
        )

        sine_mask = (
            event_mask
            & (phase == "sine")
        )

        if not np.any(
            sine_mask
        ):
            continue

        first_idx = np.flatnonzero(
            sine_mask
        )[0]

        joint_index = int(
            active_joint[
                first_idx
            ]
        )

        frequency_hz = float(
            frequencies[
                first_idx
            ]
        )

        name = str(
            joint_names[
                joint_index
            ]
        )

        t = time_values[
            sine_mask
        ]

        command = q_command[
            sine_mask,
            joint_index,
        ]

        actual = q_actual[
            sine_mask,
            joint_index,
        ]

        # ------------------------------------------------------
        # Discard first cycle to reduce transient influence
        # ------------------------------------------------------

        t_rel = (
            t - t[0]
        )

        discard_s = (
            1.0
            / frequency_hz
        )

        keep = (
            t_rel >= discard_s
        )

        if np.count_nonzero(
            keep
        ) < 20:
            keep = np.ones_like(
                t_rel,
                dtype=bool,
            )

        fit_time = t_rel[
            keep
        ]

        fit_command = command[
            keep
        ]

        fit_actual = actual[
            keep
        ]

        (
            command_amp,
            command_phase,
            command_offset,
        ) = harmonic_fit(
            fit_time,
            fit_command,
            frequency_hz,
        )

        (
            actual_amp,
            actual_phase,
            actual_offset,
        ) = harmonic_fit(
            fit_time,
            fit_actual,
            frequency_hz,
        )

        if command_amp > 1e-12:
            gain = (
                actual_amp
                / command_amp
            )
        else:
            gain = float(
                "nan"
            )

        phase_lag = (
            command_phase
            - actual_phase
        )

        phase_lag = (
            phase_lag
            + np.pi
        ) % (
            2.0
            * np.pi
        ) - np.pi

        phase_lag_deg = float(
            np.degrees(
                phase_lag
            )
        )

        equivalent_delay_s = float(
            phase_lag
            / (
                2.0
                * np.pi
                * frequency_hz
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        fit_actual
                        - fit_command
                    )
                    ** 2
                )
            )
        )

        # ------------------------------------------------------
        # Cross-joint coupling
        # ------------------------------------------------------

        other_indices = [
            idx
            for idx in range(
                q_actual.shape[1]
            )
            if idx != joint_index
        ]

        if other_indices:

            pre_mask = (
                event_mask
                & (phase == "pre")
            )

            baseline_other = np.mean(
                q_actual[
                    pre_mask,
                    :
                ],
                axis=0,
            )

            deviation = np.abs(
                q_actual[
                    sine_mask,
                    :
                ][:, other_indices]
                - baseline_other[
                    other_indices
                ]
            )

            cross_coupling_max = float(
                np.max(
                    deviation
                )
            )

        else:
            cross_coupling_max = 0.0

        rows.append(
            {
                "event_id":
                    int(event_id),

                "joint_index":
                    joint_index,

                "joint_name":
                    name,

                "frequency_hz":
                    frequency_hz,

                "command_amplitude_rad":
                    command_amp,

                "actual_amplitude_rad":
                    actual_amp,

                "gain_actual_over_command":
                    gain,

                "phase_lag_deg":
                    phase_lag_deg,

                "equivalent_delay_s":
                    equivalent_delay_s,

                "command_actual_rmse_rad":
                    rmse,

                "command_offset_rad":
                    command_offset,

                "actual_offset_rad":
                    actual_offset,

                "cross_joint_max_deviation_rad":
                    cross_coupling_max,
            }
        )

    write_csv(
        output_dir
        / "sine_event_metrics.csv",
        rows,
    )

    gains = np.asarray(
        [
            row[
                "gain_actual_over_command"
            ]
            for row in rows
        ],
        dtype=float,
    )

    phases = np.asarray(
        [
            row[
                "phase_lag_deg"
            ]
            for row in rows
        ],
        dtype=float,
    )

    summary = {
        "experiment":
            "sine",

        "event_count":
            len(rows),

        "sample_rate_hz":
            sample_rate_hz,

        "mean_gain":
            float(
                np.nanmean(
                    gains
                )
            ),

        "min_gain":
            float(
                np.nanmin(
                    gains
                )
            ),

        "max_gain":
            float(
                np.nanmax(
                    gains
                )
            ),

        "mean_phase_lag_deg":
            float(
                np.nanmean(
                    phases
                )
            ),
    }

    write_json(
        output_dir
        / "sine_summary.json",
        summary,
    )

    print(
        "[PASS] Sine analysis complete."
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    data = load_npz(
        args.input
    )

    metadata_path = (
        args.input
        .with_suffix(".json")
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file missing: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(
            f
        )

    experiment = metadata[
        "experiment"
    ]

    sample_rate_hz = float(
        metadata[
            "sample_rate_hz"
        ]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Experiment : {experiment}"
    )

    print(
        f"Input      : {args.input}"
    )

    if experiment == "static":
        analyze_static(
            data,
            args.output_dir,
            sample_rate_hz,
        )

    elif experiment == "step":
        analyze_step(
            data,
            args.output_dir,
            sample_rate_hz,
        )

    elif experiment == "sine":
        analyze_sine(
            data,
            args.output_dir,
            sample_rate_hz,
        )

    else:
        raise ValueError(
            f"Unsupported experiment: {experiment}"
        )


if __name__ == "__main__":
    main()