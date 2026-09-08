#!/usr/bin/env python3

"""
Create controlled Gate 1B corrupted ORCA demonstrations.

Supported corruption types:

    outlier
        Sparse high-amplitude joint spikes.

    dropout
        Whole-hand tracking loss modeled by zero-order hold.

    delay
        Fixed temporal latency.

The input should normally be the original clean synthetic dataset:

    data/examples/synthetic_orca_demo.npz

The script preserves the clean trajectory under:

    <trajectory_key>_clean

and replaces <trajectory_key> with the corrupted trajectory.

Example:

    python scripts/inject_orca_corruption.py \
        --input data/examples/synthetic_orca_demo.npz \
        --output data/examples/gate1b_outlier.npz \
        --type outlier
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


from research.orca_research.corruptions import (
    inject_impulsive_outliers,
    inject_temporal_delay,
    inject_tracking_dropout_hold,
)

from research.orca_research.io import (
    load_dataset,
    normalize_time,
    require_trajectory,
    save_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject controlled Gate 1B uncertainty/corruption "
            "into an ORCA trajectory."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--trajectory-key",
        default="q_command",
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=[
            "outlier",
            "dropout",
            "delay",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=11,
    )

    # ------------------------------------------------------------
    # Outlier parameters
    # ------------------------------------------------------------

    parser.add_argument(
        "--outlier-probability",
        type=float,
        default=0.003,
        help=(
            "Probability that each frame/joint sample is "
            "corrupted by an impulsive outlier."
        ),
    )

    parser.add_argument(
        "--outlier-amplitude-rad",
        type=float,
        default=0.08,
        help=(
            "Maximum absolute outlier magnitude in radians."
        ),
    )

    # ------------------------------------------------------------
    # Dropout parameters
    # ------------------------------------------------------------

    parser.add_argument(
        "--dropout-count",
        type=int,
        default=4,
        help=(
            "Number of non-overlapping tracking dropout events."
        ),
    )

    parser.add_argument(
        "--dropout-duration-s",
        type=float,
        default=0.20,
        help=(
            "Duration of each tracking dropout event."
        ),
    )

    # ------------------------------------------------------------
    # Delay parameters
    # ------------------------------------------------------------

    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.10,
        help=(
            "Injected fixed temporal delay in seconds."
        ),
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

    clean = require_trajectory(
        dataset,
        args.trajectory_key,
    ).copy()

    # Preserve the clean reference.
    dataset[
        f"{args.trajectory_key}_clean"
    ] = clean

    # Preserve the original validity signal before modifying it.
    original_tracking_valid = np.asarray(
        dataset.get(
            "tracking_valid",
            np.ones(
                clean.shape[0],
                dtype=bool,
            ),
        ),
        dtype=bool,
    )

    dataset[
        "tracking_valid_clean"
    ] = original_tracking_valid.copy()

    # ============================================================
    # 1. Impulsive outliers
    # ============================================================

    if args.type == "outlier":
        corrupted, corruption, mask = (
            inject_impulsive_outliers(
                clean,
                time_values,
                probability=args.outlier_probability,
                amplitude_rad=args.outlier_amplitude_rad,
                seed=args.seed,
            )
        )

        dataset[
            "tracking_valid"
        ] = original_tracking_valid.copy()

        dataset[
            "corruption_parameter_probability"
        ] = np.asarray(
            args.outlier_probability
        )

        dataset[
            "corruption_parameter_amplitude_rad"
        ] = np.asarray(
            args.outlier_amplitude_rad
        )

    # ============================================================
    # 2. Tracking dropout
    # ============================================================

    elif args.type == "dropout":
        (
            corrupted,
            corruption,
            mask,
            injected_tracking_valid,
        ) = inject_tracking_dropout_hold(
            clean,
            time_values,
            dropout_count=args.dropout_count,
            duration_s=args.dropout_duration_s,
            seed=args.seed,
        )

        dataset[
            "tracking_valid"
        ] = (
            original_tracking_valid
            & injected_tracking_valid
        )

        dataset[
            "corruption_parameter_dropout_count"
        ] = np.asarray(
            args.dropout_count
        )

        dataset[
            "corruption_parameter_dropout_duration_s"
        ] = np.asarray(
            args.dropout_duration_s
        )

    # ============================================================
    # 3. Temporal delay
    # ============================================================

    elif args.type == "delay":
        (
            corrupted,
            corruption,
            mask,
            delay_frames,
        ) = inject_temporal_delay(
            clean,
            time_values,
            delay_s=args.delay_s,
        )

        dataset[
            "tracking_valid"
        ] = original_tracking_valid.copy()

        dataset[
            "corruption_parameter_delay_s"
        ] = np.asarray(
            args.delay_s
        )

        dataset[
            "corruption_parameter_delay_frames"
        ] = np.asarray(
            delay_frames
        )

    else:
        raise RuntimeError(
            f"Unhandled corruption type: {args.type}"
        )

    # ------------------------------------------------------------
    # Shared output fields
    # ------------------------------------------------------------

    dataset[
        args.trajectory_key
    ] = corrupted

    dataset[
        f"{args.trajectory_key}_corruption"
    ] = corruption

    dataset[
        "corruption_mask"
    ] = mask

    dataset[
        "corruption_type"
    ] = np.asarray(
        args.type
    )

    dataset[
        "corruption_seed"
    ] = np.asarray(
        args.seed
    )

    save_npz(
        args.output,
        dataset,
    )

    affected_samples = int(
        np.count_nonzero(mask)
    )

    total_samples = int(
        mask.size
    )

    affected_ratio = (
        affected_samples
        / max(total_samples, 1)
    )

    rmse_corrupted_clean = float(
        np.sqrt(
            np.mean(
                (
                    corrupted
                    - clean
                )
                ** 2
            )
        )
    )

    print(
        f"[OK] Wrote corrupted dataset: {args.output}"
    )

    print(
        f"[INFO] Corruption type: {args.type}"
    )

    print(
        f"[INFO] Trajectory shape: {clean.shape}"
    )

    print(
        "[INFO] Affected samples: "
        f"{affected_samples}/{total_samples} "
        f"({100.0 * affected_ratio:.3f}%)"
    )

    print(
        "[INFO] Global RMSE(corrupted, clean): "
        f"{rmse_corrupted_clean:.8f} rad"
    )

    if args.type == "dropout":
        invalid_frames = int(
            np.count_nonzero(
                ~dataset["tracking_valid"]
            )
        )

        print(
            "[INFO] Tracking-invalid frames: "
            f"{invalid_frames}/{clean.shape[0]}"
        )

    if args.type == "delay":
        print(
            "[INFO] Delay frames: "
            f"{int(dataset['corruption_parameter_delay_frames'])}"
        )


if __name__ == "__main__":
    main()