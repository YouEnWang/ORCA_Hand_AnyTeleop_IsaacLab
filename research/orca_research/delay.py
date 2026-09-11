"""
Temporal-delay utilities for Gate 1B-3.

This module provides reference-based temporal alignment tools for
controlled ORCA demonstration experiments.

Positive delay convention
-------------------------
A positive delay D means:

    observed[t] ~= reference[t - D]

Therefore, after compensation:

    observed[t + D] ~= reference[t]

Important
---------
These utilities are intended primarily for OFFLINE dataset analysis
and demonstration alignment.

They do NOT remove physical latency from a realtime teleoperation
system. Online delay compensation would require prediction and/or
latency reduction.
---------
這支程式負責三件事情：

reference trajectory
       +
observed delayed trajectory
       ↓
搜尋 candidate lag
       ↓
找到最符合的 delay frames
       ↓
做 temporal alignment
"""

from __future__ import annotations

import numpy as np


def _validate_pair(
    reference: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two frame-by-joint trajectories."""

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    if reference.ndim != 2:
        raise ValueError(
            "reference must have shape (frames, joints), "
            f"got {reference.shape}"
        )

    if observed.ndim != 2:
        raise ValueError(
            "observed must have shape (frames, joints), "
            f"got {observed.shape}"
        )

    if reference.shape != observed.shape:
        raise ValueError(
            "reference and observed shapes differ: "
            f"{reference.shape} vs {observed.shape}"
        )

    if reference.shape[0] < 2:
        raise ValueError(
            "At least two frames are required."
        )

    return reference, observed


def aligned_overlap(
    reference: np.ndarray,
    observed: np.ndarray,
    delay_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return temporally aligned overlapping samples.

    For positive delay D:

        reference[:-D]
        observed[D:]

    should represent the same intended timestamps.
    """

    reference, observed = _validate_pair(
        reference,
        observed,
    )

    if delay_frames < 0:
        raise ValueError(
            "delay_frames must be >= 0"
        )

    if delay_frames >= reference.shape[0]:
        raise ValueError(
            "delay_frames is too large for trajectory."
        )

    if delay_frames == 0:
        return (
            reference.copy(),
            observed.copy(),
        )

    return (
        reference[:-delay_frames],
        observed[delay_frames:],
    )


def rmse_per_joint(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Per-joint RMSE."""

    first, second = _validate_pair(
        first,
        second,
    )

    return np.sqrt(
        np.mean(
            (first - second) ** 2,
            axis=0,
        )
    )


def rmse_global(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """One RMSE over all frames and joints."""

    first, second = _validate_pair(
        first,
        second,
    )

    return float(
        np.sqrt(
            np.mean(
                (first - second) ** 2
            )
        )
    )


def estimate_delay_frames(
    reference: np.ndarray,
    observed: np.ndarray,
    max_delay_frames: int,
    eps: float = 1e-9,
) -> tuple[int, np.ndarray]:
    """
    Estimate one global positive delay by normalized RMSE search.

    For every candidate D:

        reference[:-D]

    is compared against:

        observed[D:]

    Each joint is normalized by the standard deviation of the
    reference trajectory so that high-amplitude joints do not
    dominate the delay estimate.

    Returns
    -------
    best_delay_frames:
        Candidate delay with the smallest normalized RMSE.

    scores:
        Normalized global RMSE for candidate delays
        0 ... max_delay_frames.
    """

    reference, observed = _validate_pair(
        reference,
        observed,
    )

    if max_delay_frames < 0:
        raise ValueError(
            "max_delay_frames must be >= 0."
        )

    max_delay_frames = min(
        int(max_delay_frames),
        reference.shape[0] - 1,
    )

    scale = np.std(
        reference,
        axis=0,
    )

    scale = np.maximum(
        scale,
        eps,
    )

    scores = np.empty(
        max_delay_frames + 1,
        dtype=np.float64,
    )

    for delay in range(
        max_delay_frames + 1
    ):
        ref_segment, obs_segment = (
            aligned_overlap(
                reference,
                observed,
                delay,
            )
        )

        normalized_error = (
            obs_segment
            - ref_segment
        ) / scale

        scores[delay] = np.sqrt(
            np.mean(
                normalized_error ** 2
            )
        )

    best_delay_frames = int(
        np.argmin(
            scores
        )
    )

    return (
        best_delay_frames,
        scores,
    )


def estimate_delay_frames_per_joint(
    reference: np.ndarray,
    observed: np.ndarray,
    max_delay_frames: int,
) -> np.ndarray:
    """
    Estimate delay independently for every joint.

    Returns an integer array of shape (joints,).
    """

    reference, observed = _validate_pair(
        reference,
        observed,
    )

    max_delay_frames = min(
        int(max_delay_frames),
        reference.shape[0] - 1,
    )

    joint_count = reference.shape[1]

    output = np.zeros(
        joint_count,
        dtype=np.int64,
    )

    for joint_index in range(
        joint_count
    ):
        scores = []

        for delay in range(
            max_delay_frames + 1
        ):
            if delay == 0:
                ref_segment = reference[
                    :,
                    joint_index,
                ]

                obs_segment = observed[
                    :,
                    joint_index,
                ]

            else:
                ref_segment = reference[
                    :-delay,
                    joint_index,
                ]

                obs_segment = observed[
                    delay:,
                    joint_index,
                ]

            score = np.sqrt(
                np.mean(
                    (
                        obs_segment
                        - ref_segment
                    )
                    ** 2
                )
            )

            scores.append(
                float(score)
            )

        output[joint_index] = int(
            np.argmin(
                np.asarray(scores)
            )
        )

    return output


def build_delay_compensated(
    observed: np.ndarray,
    delay_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift a delayed trajectory earlier for OFFLINE alignment.

    For D > 0:

        corrected[:-D] = observed[D:]

    The final D samples cannot be reconstructed because their
    corresponding future observations are not available.

    Those samples are filled with NaN and marked invalid.

    Returns
    -------
    corrected:
        Shape (frames, joints).

    valid_mask:
        Shape (frames,).
        True only where compensated data are available.
    """

    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    if observed.ndim != 2:
        raise ValueError(
            "observed must have shape (frames, joints)."
        )

    if delay_frames < 0:
        raise ValueError(
            "delay_frames must be >= 0."
        )

    if delay_frames >= observed.shape[0]:
        raise ValueError(
            "delay_frames is too large."
        )

    corrected = np.full_like(
        observed,
        np.nan,
    )

    valid_mask = np.zeros(
        observed.shape[0],
        dtype=bool,
    )

    if delay_frames == 0:
        corrected[:] = observed
        valid_mask[:] = True

        return (
            corrected,
            valid_mask,
        )

    corrected[
        :-delay_frames
    ] = observed[
        delay_frames:
    ]

    valid_mask[
        :-delay_frames
    ] = True

    return (
        corrected,
        valid_mask,
    )