#!/usr/bin/env python3

"""
Compare raw and purified ORCA demonstrations.

In addition to full-trajectory metrics, this script supports
corruption-mask-aware evaluation when the raw dataset contains:

    corruption_mask

The mask-aware evaluation separates two questions:

1. Affected-region recovery
   How well does purification recover samples that were intentionally
   corrupted?

2. Clean-region distortion
   How much does purification modify samples that were NOT corrupted?

This distinction is important for uncertainty-aware demonstration
purification: a good method should repair unreliable samples while
preserving already-clean motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)


from research.orca_research.io import (
    load_dataset,
    normalize_time,
    require_trajectory,
)

from research.orca_research.metrics import (
    compute_quality_metrics,
    summarize_joint_metrics,
)

from research.orca_research.reporting import (
    plot_trajectory_overlay,
    write_csv,
    write_json,
)


def _prefix_metrics(
    prefix: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def _normalize_corruption_mask(
    mask: np.ndarray,
    trajectory_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert a corruption mask to shape (frames, joints).

    Supported input shapes:
        (frames,)
        (frames, 1)
        (frames, joints)

    A 1-D frame-level mask is broadcast to all joints.
    """

    mask = np.asarray(mask, dtype=bool)

    frame_count, joint_count = trajectory_shape

    if mask.ndim == 1:
        if mask.shape[0] != frame_count:
            raise ValueError(
                "1-D corruption_mask length does not match trajectory: "
                f"{mask.shape[0]} vs {frame_count}"
            )

        mask = np.repeat(
            mask[:, None],
            joint_count,
            axis=1,
        )

    elif mask.ndim == 2:
        if mask.shape == (frame_count, 1):
            mask = np.repeat(
                mask,
                joint_count,
                axis=1,
            )

        elif mask.shape != trajectory_shape:
            raise ValueError(
                "2-D corruption_mask shape does not match trajectory: "
                f"{mask.shape} vs {trajectory_shape}"
            )

    else:
        raise ValueError(
            "corruption_mask must be 1-D or 2-D, "
            f"got shape {mask.shape}"
        )

    return mask


def _masked_rmse_per_joint(
    estimate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate per-joint RMSE using only samples where mask == True.

    Returns NaN for a joint if no selected samples exist.
    """

    if (
        estimate.shape != reference.shape
        or estimate.shape != mask.shape
    ):
        raise ValueError(
            "Shape mismatch in masked RMSE: "
            f"estimate={estimate.shape}, "
            f"reference={reference.shape}, "
            f"mask={mask.shape}"
        )

    joint_count = estimate.shape[1]

    output = np.full(
        joint_count,
        np.nan,
        dtype=np.float64,
    )

    squared_error = (
        estimate
        - reference
    ) ** 2

    for joint_index in range(joint_count):
        selected = mask[:, joint_index]

        if np.any(selected):
            output[joint_index] = np.sqrt(
                np.mean(
                    squared_error[
                        selected,
                        joint_index,
                    ]
                )
            )

    return output


def _masked_max_abs_per_joint(
    estimate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate per-joint maximum absolute error using only mask == True.

    Returns NaN if a joint has no selected samples.
    """

    joint_count = estimate.shape[1]

    output = np.full(
        joint_count,
        np.nan,
        dtype=np.float64,
    )

    absolute_error = np.abs(
        estimate
        - reference
    )

    for joint_index in range(joint_count):
        selected = mask[:, joint_index]

        if np.any(selected):
            output[joint_index] = np.max(
                absolute_error[
                    selected,
                    joint_index,
                ]
            )

    return output


def _masked_rmse_global(
    estimate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Calculate one global RMSE over all selected frame/joint samples.
    """

    if not np.any(mask):
        return float("nan")

    error = (
        estimate
        - reference
    )[mask]

    return float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


def _recovery_percent(
    raw_rmse: np.ndarray,
    purified_rmse: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate:

        (1 - purified_rmse / raw_rmse) * 100

    only where the raw RMSE is meaningfully non-zero.
    """

    raw_rmse = np.asarray(
        raw_rmse,
        dtype=np.float64,
    )

    purified_rmse = np.asarray(
        purified_rmse,
        dtype=np.float64,
    )

    output = np.full_like(
        raw_rmse,
        np.nan,
    )

    valid = (
        np.isfinite(raw_rmse)
        & np.isfinite(purified_rmse)
        & (raw_rmse > eps)
    )

    output[valid] = (
        1.0
        - purified_rmse[valid]
        / raw_rmse[valid]
    ) * 100.0

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--purified",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/raw_vs_purified"
        ),
    )

    parser.add_argument(
        "--trajectory-key",
        default="q_command",
    )

    args = parser.parse_args()

    # ============================================================
    # Load data
    # ============================================================

    raw_dataset = load_dataset(
        args.raw
    )

    purified_dataset = load_dataset(
        args.purified
    )

    time_values = normalize_time(
        np.asarray(
            raw_dataset["time"],
            dtype=np.float64,
        )
    )

    # Raw corrupted trajectory
    raw = require_trajectory(
        raw_dataset,
        args.trajectory_key,
    )

    # Purified trajectory
    purified = require_trajectory(
        purified_dataset,
        args.trajectory_key,
    )

    # Known clean ground truth from controlled corruption experiments
    clean = require_trajectory(
        raw_dataset,
        f"{args.trajectory_key}_clean",
    )

    if (
        raw.shape != purified.shape
        or raw.shape != clean.shape
    ):
        raise ValueError(
            "Trajectory shape mismatch: "
            f"raw={raw.shape}, "
            f"purified={purified.shape}, "
            f"clean={clean.shape}"
        )

    # ============================================================
    # Whole-trajectory signal metrics
    # ============================================================

    raw_metrics = compute_quality_metrics(
        time_values,
        raw,
    )

    purified_metrics = compute_quality_metrics(
        time_values,
        purified,
    )

    # ============================================================
    # Whole-trajectory ground-truth RMSE
    # ============================================================

    delta: dict[str, object] = {
        # How much purification changes the raw trajectory.
        "rmse_raw_minus_purified":
            np.sqrt(
                np.mean(
                    (raw - purified) ** 2,
                    axis=0,
                )
            ),

        # How far corrupted raw data is from clean ground truth.
        "rmse_raw_minus_clean":
            np.sqrt(
                np.mean(
                    (raw - clean) ** 2,
                    axis=0,
                )
            ),

        # How far purified data is from clean ground truth.
        "rmse_purified_minus_clean":
            np.sqrt(
                np.mean(
                    (purified - clean) ** 2,
                    axis=0,
                )
            ),
    }

    # ============================================================
    # Corruption-mask-aware evaluation
    # ============================================================

    mask_metrics: dict[str, object] = {
        "corruption_mask_available": False,
    }

    if "corruption_mask" in raw_dataset:

        corruption_mask = _normalize_corruption_mask(
            raw_dataset["corruption_mask"],
            raw.shape,
        )

        clean_mask = ~corruption_mask

        # --------------------------------------------------------
        # Per-joint corruption statistics
        # --------------------------------------------------------

        affected_count = np.sum(
            corruption_mask,
            axis=0,
        )

        affected_fraction = np.mean(
            corruption_mask,
            axis=0,
        )

        # --------------------------------------------------------
        # Affected-region recovery
        #
        # Only samples deliberately corrupted by Gate 1B.
        # --------------------------------------------------------

        rmse_raw_affected = (
            _masked_rmse_per_joint(
                raw,
                clean,
                corruption_mask,
            )
        )

        rmse_purified_affected = (
            _masked_rmse_per_joint(
                purified,
                clean,
                corruption_mask,
            )
        )

        recovery_percent = (
            _recovery_percent(
                rmse_raw_affected,
                rmse_purified_affected,
            )
        )

        # --------------------------------------------------------
        # Clean-region distortion
        #
        # Samples that were NOT corrupted.
        # This measures collateral modification caused by the
        # purification method itself.
        # --------------------------------------------------------

        rmse_raw_unaffected = (
            _masked_rmse_per_joint(
                raw,
                clean,
                clean_mask,
            )
        )

        rmse_purified_unaffected = (
            _masked_rmse_per_joint(
                purified,
                clean,
                clean_mask,
            )
        )

        max_abs_purified_unaffected = (
            _masked_max_abs_per_joint(
                purified,
                clean,
                clean_mask,
            )
        )

        # --------------------------------------------------------
        # Global mask-aware values
        # --------------------------------------------------------

        affected_global_raw = (
            _masked_rmse_global(
                raw,
                clean,
                corruption_mask,
            )
        )

        affected_global_purified = (
            _masked_rmse_global(
                purified,
                clean,
                corruption_mask,
            )
        )

        unaffected_global_raw = (
            _masked_rmse_global(
                raw,
                clean,
                clean_mask,
            )
        )

        unaffected_global_purified = (
            _masked_rmse_global(
                purified,
                clean,
                clean_mask,
            )
        )

        if (
            np.isfinite(
                affected_global_raw
            )
            and affected_global_raw > 1e-12
        ):
            affected_global_recovery_percent = (
                1.0
                - affected_global_purified
                / affected_global_raw
            ) * 100.0
        else:
            affected_global_recovery_percent = float(
                "nan"
            )

        mask_metrics = {
            "corruption_mask_available":
                True,

            # Dataset-level mask information
            "corruption_affected_sample_count_total":
                int(
                    np.sum(
                        corruption_mask
                    )
                ),

            "corruption_affected_sample_fraction_total":
                float(
                    np.mean(
                        corruption_mask
                    )
                ),

            # Per-joint mask information
            "corruption_affected_count":
                affected_count,

            "corruption_affected_fraction":
                affected_fraction,

            # Per-joint affected-region recovery
            "rmse_raw_minus_clean_affected":
                rmse_raw_affected,

            "rmse_purified_minus_clean_affected":
                rmse_purified_affected,

            "affected_recovery_percent":
                recovery_percent,

            # Per-joint clean-region distortion
            "rmse_raw_minus_clean_unaffected":
                rmse_raw_unaffected,

            "rmse_purified_minus_clean_unaffected":
                rmse_purified_unaffected,

            "max_abs_purified_minus_clean_unaffected":
                max_abs_purified_unaffected,

            # Global affected-region results
            "rmse_raw_minus_clean_affected_global":
                affected_global_raw,

            "rmse_purified_minus_clean_affected_global":
                affected_global_purified,

            "affected_recovery_percent_global":
                affected_global_recovery_percent,

            # Global clean-region results
            "rmse_raw_minus_clean_unaffected_global":
                unaffected_global_raw,

            "rmse_purified_minus_clean_unaffected_global":
                unaffected_global_purified,
        }

    else:
        print(
            "[WARN] raw dataset has no corruption_mask; "
            "mask-aware metrics will be skipped."
        )

    # ============================================================
    # Combine metrics
    # ============================================================

    all_metrics: dict[str, object] = {}

    all_metrics.update(
        _prefix_metrics(
            "raw",
            raw_metrics,
        )
    )

    all_metrics.update(
        _prefix_metrics(
            "purified",
            purified_metrics,
        )
    )

    all_metrics.update(
        delta
    )

    all_metrics.update(
        mask_metrics
    )

    # ============================================================
    # Reporting
    # ============================================================

    joint_names = list(
        raw_dataset.get(
            "joint_names",
            [],
        )
    )

    rows = summarize_joint_metrics(
        joint_names,
        all_metrics,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        args.output_dir
        / "raw_vs_purified_joint_metrics.csv",
        rows,
    )

    write_json(
        args.output_dir
        / "raw_vs_purified_summary.json",
        all_metrics,
    )

    plot_trajectory_overlay(
        args.output_dir
        / "raw_vs_purified_preview.png",
        time_values,
        raw,
        purified,
        joint_names,
    )

    # ============================================================
    # Terminal summary
    # ============================================================

    print(
        f"Wrote comparison to "
        f"{args.output_dir}"
    )

    if (
        mask_metrics[
            "corruption_mask_available"
        ]
    ):
        print(
            "[MASK] affected samples: "
            f"{mask_metrics['corruption_affected_sample_count_total']}"
        )

        print(
            "[MASK] affected fraction: "
            f"{100.0 * mask_metrics['corruption_affected_sample_fraction_total']:.4f}%"
        )

        print(
            "[MASK] affected RMSE raw-clean: "
            f"{mask_metrics['rmse_raw_minus_clean_affected_global']:.8f} rad"
        )

        print(
            "[MASK] affected RMSE purified-clean: "
            f"{mask_metrics['rmse_purified_minus_clean_affected_global']:.8f} rad"
        )

        print(
            "[MASK] affected recovery: "
            f"{mask_metrics['affected_recovery_percent_global']:.2f}%"
        )

        print(
            "[MASK] clean-region distortion "
            "(purified-clean RMSE): "
            f"{mask_metrics['rmse_purified_minus_clean_unaffected_global']:.8f} rad"
        )


if __name__ == "__main__":
    main()