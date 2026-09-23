#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human-to-ORCA task-space vector alignment utilities.

Purpose
-------
This module isolates the coordinate-alignment logic used by the ORCA
AnyTeleop pipeline.  It is intentionally independent of MediaPipe,
UDP, OpenCV, and Isaac Lab.

The teleoperation server produces human task-space vectors in the
coordinate system returned by the pinned dex-retargeting MediaPipe
example.  Before those vectors are passed to dex-retargeting's
VectorOptimizer, they are rotated and uniformly scaled into the ORCA
retargeting frame.

Alignment convention
--------------------
For a single column vector::

    v_orca = scale * R @ v_human

The server stores K vectors row-wise in an array with shape (K, 3), so
NumPy uses the equivalent form::

    V_orca = scale * V_human @ R.T

This module deliberately performs only this rigid rotation + uniform
scale.  No temporal filtering, uncertainty weighting, or retargeting is
performed here.  Keeping this step isolated makes Gate 3A analysis able
to distinguish perception/vector jitter from later optimizer effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_vector_alignment(
    alignment_path: Path,
):
    """
    Load and validate a Human -> ORCA vector-frame alignment YAML.

    Parameters
    ----------
    alignment_path:
        Path to a YAML file containing:

        - ``human_to_orca_rotation``: 3 x 3 proper rotation matrix.
        - ``scale``: positive finite scalar.

    Returns
    -------
    tuple
        ``(rotation, scale, determinant, orthogonality_error)`` where:

        - ``rotation`` is ``float32`` with shape ``(3, 3)``.
        - ``scale`` is a Python ``float``.
        - ``determinant`` is ``det(rotation)``.
        - ``orthogonality_error`` is the maximum element-wise error of
          ``R @ R.T - I``.

    Notes
    -----
    The validation thresholds are intentionally kept identical to the
    pre-refactor server so this refactor does not change experiment
    behavior.
    """

    alignment_path = alignment_path.expanduser().resolve()

    if not alignment_path.exists():
        raise FileNotFoundError(
            "Alignment YAML does not exist:\n"
            f"{alignment_path}"
        )

    with open(
        alignment_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if "human_to_orca_rotation" not in data:
        raise KeyError(
            "Alignment YAML is missing "
            "'human_to_orca_rotation'."
        )

    if "scale" not in data:
        raise KeyError(
            "Alignment YAML is missing 'scale'."
        )

    rotation = np.asarray(
        data["human_to_orca_rotation"],
        dtype=np.float32,
    )
    scale = float(data["scale"])

    if rotation.shape != (3, 3):
        raise RuntimeError(
            "Alignment rotation must have shape (3, 3), "
            f"got {rotation.shape}."
        )

    if not np.all(np.isfinite(rotation)):
        raise RuntimeError(
            "Alignment rotation contains NaN/Inf."
        )

    if not np.isfinite(scale):
        raise RuntimeError(
            "Alignment scale is NaN/Inf."
        )

    if scale <= 0.0:
        raise RuntimeError(
            "Alignment scale must be positive."
        )

    determinant = float(np.linalg.det(rotation))

    orthogonality_error = float(
        np.max(
            np.abs(
                rotation
                @ rotation.T
                - np.eye(3, dtype=np.float32)
            )
        )
    )

    if abs(determinant - 1.0) > 0.02:
        raise RuntimeError(
            "Alignment rotation is not a proper rotation:\n"
            f"det(R)={determinant}"
        )

    if orthogonality_error > 0.02:
        raise RuntimeError(
            "Alignment rotation is not sufficiently orthogonal:\n"
            f"error={orthogonality_error}"
        )

    return (
        rotation,
        scale,
        determinant,
        orthogonality_error,
    )


def apply_vector_alignment(
    reference_vectors_raw: np.ndarray,
    rotation: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Transform row-wise human vectors into the ORCA task-space frame.

    Parameters
    ----------
    reference_vectors_raw:
        Human vectors with shape ``(K, 3)``.
    rotation:
        3 x 3 Human -> ORCA rotation matrix.
    scale:
        Positive uniform scale.

    Returns
    -------
    numpy.ndarray
        Aligned vectors with shape ``(K, 3)`` and dtype ``float32``.

    Important
    ---------
    The selected dex-retargeting YAML must keep ``scaling_factor: 1.0``
    when this external alignment scale is applied.  Otherwise scale would
    be applied twice.
    """

    aligned = scale * (
        reference_vectors_raw @ rotation.T
    )

    return np.asarray(
        aligned,
        dtype=np.float32,
    )
