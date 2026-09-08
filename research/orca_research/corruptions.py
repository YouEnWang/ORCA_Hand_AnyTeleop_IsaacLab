"""
Controlled trajectory corruptions for Gate 1B experiments.

This module extends the Gate 1A synthetic jitter benchmark with three
additional controlled corruption types:

1. Impulsive outliers
   - Simulates occasional large pose / retargeting errors.
   - Individual joint samples receive short high-amplitude spikes.

2. Tracking dropout with zero-order hold
   - Simulates temporary loss of vision-based hand tracking.
   - During dropout, the previous valid robot command is held.

3. Temporal delay
   - Simulates end-to-end latency between the original intended trajectory
     and the command received by the downstream system.

All corruption functions preserve the original trajectory externally so that
Gate 1B can compare the corrupted trajectory against the known clean ground
truth.

Important:
    These functions are intended for controlled offline experiments.
    They do NOT claim to reproduce the exact statistical distribution of
    real MediaPipe / RealSense failures.
"""

from __future__ import annotations

import numpy as np


def _validate_trajectory(
    values: np.ndarray,
    time_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize trajectory inputs."""

    values = np.asarray(values, dtype=np.float64)
    time_values = np.asarray(time_values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(
            f"values must have shape (frames, joints), got {values.shape}"
        )

    if time_values.ndim != 1:
        raise ValueError(
            f"time_values must be 1-D, got {time_values.shape}"
        )

    if len(time_values) != values.shape[0]:
        raise ValueError(
            "time_values and values must have the same number of frames: "
            f"{len(time_values)} vs {values.shape[0]}"
        )

    if len(time_values) < 2:
        raise ValueError("At least two trajectory frames are required.")

    dt = np.diff(time_values)

    if np.any(dt <= 0.0):
        raise ValueError("time_values must be strictly increasing.")

    return values, time_values


def inject_impulsive_outliers(
    values: np.ndarray,
    time_values: np.ndarray,
    probability: float = 0.003,
    amplitude_rad: float = 0.08,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject sparse impulsive joint outliers.

    Each frame/joint sample independently has `probability` chance of being
    corrupted by an impulse. The impulse magnitude is randomly chosen between
    50% and 100% of `amplitude_rad`, with a random sign.

    Parameters
    ----------
    values:
        Clean trajectory, shape (frames, joints).

    time_values:
        Trajectory timestamps. Used only for validation here.

    probability:
        Probability that one frame/joint sample becomes an outlier.

    amplitude_rad:
        Maximum absolute impulse magnitude in radians.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    corrupted:
        Corrupted trajectory.

    corruption:
        Additive corruption such that:
            corrupted = values + corruption

    mask:
        Boolean array with shape (frames, joints).
        True indicates an injected outlier.
    """

    values, time_values = _validate_trajectory(
        values,
        time_values,
    )

    if not (0.0 <= probability <= 1.0):
        raise ValueError(
            f"probability must be in [0, 1], got {probability}"
        )

    if amplitude_rad < 0.0:
        raise ValueError(
            f"amplitude_rad must be non-negative, got {amplitude_rad}"
        )

    rng = np.random.default_rng(seed)

    mask = rng.random(values.shape) < probability

    sign = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=values.shape,
    )

    magnitude = rng.uniform(
        low=0.5 * amplitude_rad,
        high=amplitude_rad,
        size=values.shape,
    )

    corruption = np.zeros_like(values)

    corruption[mask] = (
        sign[mask]
        * magnitude[mask]
    )

    corrupted = values + corruption

    return corrupted, corruption, mask


def inject_tracking_dropout_hold(
    values: np.ndarray,
    time_values: np.ndarray,
    dropout_count: int = 4,
    duration_s: float = 0.20,
    seed: int = 13,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Inject temporary whole-hand tracking dropouts.

    A dropout is modeled as a zero-order hold:
    the last valid command immediately before the dropout is held until
    tracking becomes valid again.

    This is a reasonable command-level abstraction of a teleoperation safety
    strategy such as "hold last valid pose" during short vision failures.

    Returns
    -------
    corrupted:
        Trajectory after dropout/hold corruption.

    corruption:
        Difference:
            corrupted - clean

    mask:
        Boolean array with shape (frames, joints).
        True indicates samples affected by tracking dropout.

    tracking_valid:
        Boolean array with shape (frames,).
        False during injected dropout windows.
    """

    values, time_values = _validate_trajectory(
        values,
        time_values,
    )

    if dropout_count < 0:
        raise ValueError(
            f"dropout_count must be >= 0, got {dropout_count}"
        )

    if duration_s <= 0.0:
        raise ValueError(
            f"duration_s must be positive, got {duration_s}"
        )

    dt = float(
        np.median(
            np.diff(time_values)
        )
    )

    duration_frames = max(
        1,
        int(round(duration_s / dt)),
    )

    frame_count = values.shape[0]
    joint_count = values.shape[1]

    if duration_frames >= frame_count:
        raise ValueError(
            "dropout duration is too long for this trajectory: "
            f"{duration_frames} frames >= {frame_count}"
        )

    corrupted = values.copy()

    mask = np.zeros(
        (frame_count, joint_count),
        dtype=bool,
    )

    tracking_valid = np.ones(
        frame_count,
        dtype=bool,
    )

    if dropout_count == 0:
        return (
            corrupted,
            corrupted - values,
            mask,
            tracking_valid,
        )

    rng = np.random.default_rng(seed)

    # Leave frame 0 valid because it is used as the hold reference.
    candidate_starts = np.arange(
        1,
        frame_count - duration_frames,
    )

    rng.shuffle(candidate_starts)

    chosen_starts: list[int] = []

    for candidate in candidate_starts:
        candidate_end = candidate + duration_frames

        overlap = False

        for existing in chosen_starts:
            existing_end = existing + duration_frames

            if (
                candidate < existing_end
                and existing < candidate_end
            ):
                overlap = True
                break

        if overlap:
            continue

        chosen_starts.append(
            int(candidate)
        )

        if len(chosen_starts) >= dropout_count:
            break

    if len(chosen_starts) < dropout_count:
        raise RuntimeError(
            "Could not place the requested number of non-overlapping "
            f"dropout windows: requested={dropout_count}, "
            f"placed={len(chosen_starts)}"
        )

    chosen_starts.sort()

    for start in chosen_starts:
        end = min(
            start + duration_frames,
            frame_count,
        )

        hold_value = corrupted[
            start - 1
        ].copy()

        corrupted[
            start:end,
            :
        ] = hold_value

        mask[
            start:end,
            :
        ] = True

        tracking_valid[
            start:end
        ] = False

    corruption = corrupted - values

    return (
        corrupted,
        corruption,
        mask,
        tracking_valid,
    )


def inject_temporal_delay(
    values: np.ndarray,
    time_values: np.ndarray,
    delay_s: float = 0.10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Inject a fixed temporal delay.

    The output command at time t corresponds to the clean command from
    approximately t - delay_s.

    Initial frames that do not have a previous source sample are filled with
    the first clean pose.

    Returns
    -------
    corrupted:
        Delayed trajectory.

    corruption:
        Difference:
            corrupted - clean

    mask:
        Boolean array with shape (frames, joints).
        True where the delayed value differs from the clean trajectory.

    delay_frames:
        Delay expressed as an integer number of samples.
    """

    values, time_values = _validate_trajectory(
        values,
        time_values,
    )

    if delay_s < 0.0:
        raise ValueError(
            f"delay_s must be non-negative, got {delay_s}"
        )

    dt = float(
        np.median(
            np.diff(time_values)
        )
    )

    delay_frames = int(
        round(delay_s / dt)
    )

    if delay_frames <= 0:
        corrupted = values.copy()

        return (
            corrupted,
            corrupted - values,
            np.zeros_like(
                values,
                dtype=bool,
            ),
            0,
        )

    if delay_frames >= values.shape[0]:
        raise ValueError(
            "delay is longer than the complete trajectory: "
            f"{delay_frames} frames >= {values.shape[0]}"
        )

    corrupted = np.empty_like(values)

    corrupted[
        :delay_frames,
        :
    ] = values[
        0,
        :
    ]

    corrupted[
        delay_frames:,
        :
    ] = values[
        :-delay_frames,
        :
    ]

    corruption = corrupted - values

    mask = np.abs(
        corruption
    ) > 1e-12

    return (
        corrupted,
        corruption,
        mask,
        delay_frames,
    )