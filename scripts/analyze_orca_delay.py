#!/usr/bin/env python3

"""
Analyze temporal delay in controlled ORCA demonstrations.

Gate 1B-3 focuses on temporal alignment rather than only signal
smoothness.

The script performs:

1. Global delay estimation.
2. Per-joint delay estimation.
3. Raw trajectory RMSE against clean ground truth.
4. Oracle alignment using known injected delay.
5. Estimated alignment using the automatically estimated delay.
6. Candidate-delay error curve generation.
7. Clean / delayed / aligned trajectory visualization.

Typical Gate 1B-3 usage:

    python scripts/analyze_orca_delay.py \
        --input data/examples/gate1b_delay_raw.npz \
        --output-dir results/gate1b/delay/delay_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    ),
)


from research.orca_research.delay import (
    aligned_overlap,
    build_delay_compensated,
    estimate_delay_frames,
    estimate_delay_frames_per_joint,
)

from research.orca_research.io import (
    load_dataset,
    normalize_time,
    require_trajectory,
)

from research.orca_research.metrics import (
    estimate_sample_rate_hz,
)

from research.orca_research.reporting import (
    write_csv,
    write_json,
)


def per_joint_rmse(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        np.mean(
            (first - second) ** 2,
            axis=0,
        )
    )


def global_rmse(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                (first - second) ** 2
            )
        )
    )


def evaluate_alignment(
    clean: np.ndarray,
    observed: np.ndarray,
    delay_frames: int,
) -> dict[str, object]:
    """
    Compare raw and delay-compensated trajectories on the same
    reference interval.

    For delay D, the common reference interval is:

        clean[:-D]

    Raw baseline:
        observed[:-D]

    Compensated:
        observed[D:]
    """

    if delay_frames == 0:
        reference = clean
        raw_same_interval = observed
        aligned = observed

    else:
        reference = clean[
            :-delay_frames
        ]

        raw_same_interval = observed[
            :-delay_frames
        ]

        aligned = observed[
            delay_frames:
        ]

    raw_joint_rmse = per_joint_rmse(
        raw_same_interval,
        reference,
    )

    aligned_joint_rmse = per_joint_rmse(
        aligned,
        reference,
    )

    raw_global_rmse = global_rmse(
        raw_same_interval,
        reference,
    )

    aligned_global_rmse = global_rmse(
        aligned,
        reference,
    )

    if raw_global_rmse > 1e-12:
        recovery_percent = (
            1.0
            - aligned_global_rmse
            / raw_global_rmse
        ) * 100.0
    else:
        recovery_percent = float(
            "nan"
        )

    joint_recovery = np.full(
        raw_joint_rmse.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        raw_joint_rmse > 1e-12
    )

    joint_recovery[valid] = (
        1.0
        - aligned_joint_rmse[valid]
        / raw_joint_rmse[valid]
    ) * 100.0

    return {
        "raw_rmse_per_joint":
            raw_joint_rmse,

        "aligned_rmse_per_joint":
            aligned_joint_rmse,

        "recovery_percent_per_joint":
            joint_recovery,

        "raw_rmse_global":
            raw_global_rmse,

        "aligned_rmse_global":
            aligned_global_rmse,

        "recovery_percent_global":
            recovery_percent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal delay and temporal alignment "
            "for Gate 1B-3."
        )
    )

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

    parser.add_argument(
        "--reference-key",
        default="q_desired",
        help=(
            "Upstream/intended trajectory used for "
            "delay estimation."
        ),
    )

    parser.add_argument(
        "--observed-key",
        default="q_command",
        help=(
            "Trajectory suspected of being delayed."
        ),
    )

    parser.add_argument(
        "--clean-key",
        default="q_command_clean",
        help=(
            "Known clean ground truth used only for "
            "controlled evaluation."
        ),
    )

    parser.add_argument(
        "--max-delay-s",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.input
    )

    time_values = normalize_time(
        np.asarray(
            dataset["time"],
            dtype=np.float64,
        )
    )

    reference = require_trajectory(
        dataset,
        args.reference_key,
    )

    observed = require_trajectory(
        dataset,
        args.observed_key,
    )

    if args.clean_key in dataset:
        clean = require_trajectory(
            dataset,
            args.clean_key,
        )
    else:
        print(
            "[WARN] clean-key not found; "
            "using reference as clean evaluation trajectory."
        )

        clean = reference.copy()

    if (
        reference.shape != observed.shape
        or reference.shape != clean.shape
    ):
        raise ValueError(
            "Trajectory shape mismatch: "
            f"reference={reference.shape}, "
            f"observed={observed.shape}, "
            f"clean={clean.shape}"
        )

    sample_rate_hz = (
        estimate_sample_rate_hz(
            time_values
        )
    )

    max_delay_frames = int(
        round(
            args.max_delay_s
            * sample_rate_hz
        )
    )

    # ------------------------------------------------------------
    # Known injected delay
    # ------------------------------------------------------------

    known_delay_frames = None

    if (
        "corruption_parameter_delay_frames"
        in dataset
    ):
        known_delay_frames = int(
            np.asarray(
                dataset[
                    "corruption_parameter_delay_frames"
                ]
            ).item()
        )

    # ------------------------------------------------------------
    # Estimate global delay
    # ------------------------------------------------------------

    (
        estimated_delay_frames,
        candidate_scores,
    ) = estimate_delay_frames(
        reference,
        observed,
        max_delay_frames=max_delay_frames,
    )

    estimated_delay_s = (
        estimated_delay_frames
        / sample_rate_hz
    )

    # ------------------------------------------------------------
    # Per-joint estimates
    # ------------------------------------------------------------

    estimated_delay_per_joint = (
        estimate_delay_frames_per_joint(
            reference,
            observed,
            max_delay_frames=max_delay_frames,
        )
    )

    # ------------------------------------------------------------
    # Full unaligned RMSE
    # ------------------------------------------------------------

    raw_full_rmse_per_joint = (
        per_joint_rmse(
            observed,
            clean,
        )
    )

    raw_full_rmse_global = (
        global_rmse(
            observed,
            clean,
        )
    )

    # ------------------------------------------------------------
    # Estimated-delay alignment evaluation
    # ------------------------------------------------------------

    estimated_metrics = (
        evaluate_alignment(
            clean,
            observed,
            estimated_delay_frames,
        )
    )

    # ------------------------------------------------------------
    # Known-delay / oracle evaluation
    # ------------------------------------------------------------

    known_metrics = None

    if known_delay_frames is not None:
        known_metrics = (
            evaluate_alignment(
                clean,
                observed,
                known_delay_frames,
            )
        )

    # ------------------------------------------------------------
    # Build compensated trajectory for plotting
    # ------------------------------------------------------------

    (
        estimated_compensated,
        compensated_valid,
    ) = build_delay_compensated(
        observed,
        estimated_delay_frames,
    )

    # ------------------------------------------------------------
    # Joint metrics table
    # ------------------------------------------------------------

    joint_names = list(
        dataset.get(
            "joint_names",
            [],
        )
    )

    rows = []

    for joint_index in range(
        observed.shape[1]
    ):
        name = (
            joint_names[joint_index]
            if joint_index < len(joint_names)
            else f"joint_{joint_index:02d}"
        )

        row = {
            "joint":
                name,

            "estimated_delay_frames":
                int(
                    estimated_delay_per_joint[
                        joint_index
                    ]
                ),

            "estimated_delay_s":
                float(
                    estimated_delay_per_joint[
                        joint_index
                    ]
                    / sample_rate_hz
                ),

            "raw_full_rmse":
                float(
                    raw_full_rmse_per_joint[
                        joint_index
                    ]
                ),

            "estimated_alignment_raw_rmse":
                float(
                    estimated_metrics[
                        "raw_rmse_per_joint"
                    ][joint_index]
                ),

            "estimated_alignment_rmse":
                float(
                    estimated_metrics[
                        "aligned_rmse_per_joint"
                    ][joint_index]
                ),

            "estimated_alignment_recovery_percent":
                float(
                    estimated_metrics[
                        "recovery_percent_per_joint"
                    ][joint_index]
                ),
        }

        if known_metrics is not None:
            row.update(
                {
                    "known_alignment_raw_rmse":
                        float(
                            known_metrics[
                                "raw_rmse_per_joint"
                            ][joint_index]
                        ),

                    "known_alignment_rmse":
                        float(
                            known_metrics[
                                "aligned_rmse_per_joint"
                            ][joint_index]
                        ),

                    "known_alignment_recovery_percent":
                        float(
                            known_metrics[
                                "recovery_percent_per_joint"
                            ][joint_index]
                        ),
                }
            )

        rows.append(
            row
        )

    # ------------------------------------------------------------
    # Candidate delay curve
    # ------------------------------------------------------------

    candidate_rows = []

    for delay_frames, score in enumerate(
        candidate_scores
    ):
        candidate_rows.append(
            {
                "delay_frames":
                    delay_frames,

                "delay_s":
                    delay_frames
                    / sample_rate_hz,

                "normalized_rmse_score":
                    float(score),
            }
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary: dict[str, object] = {
        "sample_rate_hz":
            sample_rate_hz,

        "max_delay_search_s":
            args.max_delay_s,

        "max_delay_search_frames":
            max_delay_frames,

        "known_delay_frames":
            known_delay_frames,

        "known_delay_s":
            (
                known_delay_frames
                / sample_rate_hz
                if known_delay_frames
                is not None
                else None
            ),

        "estimated_delay_frames":
            estimated_delay_frames,

        "estimated_delay_s":
            estimated_delay_s,

        "estimated_delay_per_joint_frames":
            estimated_delay_per_joint,

        "estimated_delay_per_joint_s":
            (
                estimated_delay_per_joint
                / sample_rate_hz
            ),

        "raw_full_rmse_per_joint":
            raw_full_rmse_per_joint,

        "raw_full_rmse_global":
            raw_full_rmse_global,

        "estimated_alignment_raw_rmse_global":
            estimated_metrics[
                "raw_rmse_global"
            ],

        "estimated_alignment_rmse_global":
            estimated_metrics[
                "aligned_rmse_global"
            ],

        "estimated_alignment_recovery_percent_global":
            estimated_metrics[
                "recovery_percent_global"
            ],
    }

    if known_metrics is not None:
        summary.update(
            {
                "known_alignment_raw_rmse_global":
                    known_metrics[
                        "raw_rmse_global"
                    ],

                "known_alignment_rmse_global":
                    known_metrics[
                        "aligned_rmse_global"
                    ],

                "known_alignment_recovery_percent_global":
                    known_metrics[
                        "recovery_percent_global"
                    ],

                "delay_estimation_error_frames":
                    (
                        estimated_delay_frames
                        - known_delay_frames
                    ),

                "delay_estimation_error_s":
                    (
                        estimated_delay_frames
                        - known_delay_frames
                    )
                    / sample_rate_hz,
            }
        )

    # ------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        args.output_dir
        / "delay_analysis_summary.json",
        summary,
    )

    write_csv(
        args.output_dir
        / "delay_joint_metrics.csv",
        rows,
    )

    write_csv(
        args.output_dir
        / "delay_search_curve.csv",
        candidate_rows,
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        joint_count = min(
            observed.shape[1],
            6,
        )

        fig, axes = plt.subplots(
            joint_count,
            1,
            figsize=(
                10,
                1.9 * joint_count,
            ),
            sharex=True,
        )

        if joint_count == 1:
            axes = [axes]

        for joint_index, axis in enumerate(
            axes
        ):
            name = (
                joint_names[joint_index]
                if joint_index
                < len(joint_names)
                else f"joint_{joint_index:02d}"
            )

            axis.plot(
                time_values,
                clean[:, joint_index],
                label="clean",
                linewidth=1.2,
            )

            axis.plot(
                time_values,
                observed[:, joint_index],
                label="delayed",
                linewidth=1.0,
            )

            axis.plot(
                time_values,
                estimated_compensated[
                    :,
                    joint_index,
                ],
                label="aligned",
                linewidth=1.0,
            )

            axis.set_ylabel(
                name,
                fontsize=8,
            )

            axis.grid(
                True,
                alpha=0.25,
            )

        axes[-1].set_xlabel(
            "time (s)"
        )

        axes[0].legend(
            loc="upper right"
        )

        fig.tight_layout()

        fig.savefig(
            args.output_dir
            / "delay_alignment_preview.png",
            dpi=160,
        )

        plt.close(fig)

    except ImportError:
        print(
            "[WARN] matplotlib unavailable; "
            "delay plot skipped."
        )

    # ------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------

    print(
        "[OK] Delay analysis complete."
    )

    if known_delay_frames is not None:
        print(
            "[DELAY] Known delay: "
            f"{known_delay_frames} frames "
            f"({known_delay_frames / sample_rate_hz:.6f} s)"
        )

    print(
        "[DELAY] Estimated delay: "
        f"{estimated_delay_frames} frames "
        f"({estimated_delay_s:.6f} s)"
    )

    if known_delay_frames is not None:
        print(
            "[DELAY] Estimation error: "
            f"{estimated_delay_frames - known_delay_frames} frames"
        )

    print(
        "[RMSE] Raw full trajectory: "
        f"{raw_full_rmse_global:.8f} rad"
    )

    print(
        "[RMSE] Estimated-aligned overlap: "
        f"{estimated_metrics['aligned_rmse_global']:.8f} rad"
    )

    print(
        "[RECOVERY] Estimated alignment: "
        f"{estimated_metrics['recovery_percent_global']:.2f}%"
    )

    print(
        f"[OK] Results written to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()