#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gate 3A JSONL research recorder for the ORCA AnyTeleop pipeline.

Purpose
-------
This module isolates experiment recording from control/network logic.
Each captured camera frame can therefore be recorded even when UDP is
fully disabled with ``--dry-run``.

The recorder preserves the schema used by the successful Gate 3A smoke
baseline.  Valid frames store the complete analysis chain:

    joint_pos                  (21, 3)
        -> reference_vectors_raw       (K, 3)
        -> reference_vectors_aligned   (K, 3)
        -> positions_rad               (17,)

``keypoint_2d`` is also stored as a numeric ``(21, 3)`` array after
converting MediaPipe's ``NormalizedLandmarkList`` protobuf object.

Invalid detection frames remain in the JSONL stream with
``tracking_valid=False`` and ``None`` for unavailable data.  This is
required for later dropout-duration and tracking-recovery analysis.

This module intentionally does NOT filter, interpolate, smooth, or
repair data.  Gate 3A is the raw perception/task-space baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def mediapipe_landmark_list_to_array(
    landmarks,
):
    """
    Convert MediaPipe ``NormalizedLandmarkList`` to ``(21, 3)``.

    Parameters
    ----------
    landmarks:
        Usually a MediaPipe protobuf object exposing ``.landmark``.  A
        NumPy/list fallback is retained for compatibility with future
        detector wrappers.

    Returns
    -------
    numpy.ndarray or None
        ``float32`` array with columns ``[x, y, z]`` and shape ``(21, 3)``.

    Raises
    ------
    RuntimeError
        If the object cannot be represented as exactly 21 3-D landmarks.
    """

    if landmarks is None:
        return None

    if hasattr(landmarks, "landmark"):
        array = np.asarray(
            [
                [
                    float(lm.x),
                    float(lm.y),
                    float(lm.z),
                ]
                for lm in landmarks.landmark
            ],
            dtype=np.float32,
        )
    else:
        array = np.asarray(
            landmarks,
            dtype=np.float32,
        )

    if array.shape != (21, 3):
        raise RuntimeError(
            "Expected MediaPipe keypoints with shape (21, 3), "
            f"got {array.shape}"
        )

    return array


class Gate3ARecorder:
    """
    Thin JSONL writer used by ``orca_anyteleop_server.py``.

    ``path=None`` disables recording while keeping call sites simple.
    The file is opened in ``"w"`` mode exactly like the pre-refactor
    server, so rerunning the same experiment path overwrites the old file.
    """

    def __init__(
        self,
        path: Optional[Path],
    ) -> None:
        self.path = path
        self._file = None

        if path is not None:
            resolved = path.expanduser().resolve()
            resolved.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Preserve the previous relative-path opening behavior while
            # creating the resolved parent directory above.
            self._file = path.expanduser().open(
                "w",
                encoding="utf-8",
            )

            print("Recording JSONL   :", path)

    @property
    def enabled(self) -> bool:
        """Return True when a JSONL output file is open."""

        return self._file is not None

    def _write(self, record: dict) -> None:
        if self._file is None:
            return

        self._file.write(
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        self._file.write("\n")

    def write_invalid_detection(
        self,
        *,
        seq: int,
        timestamp: float,
        capture_monotonic_s: float,
    ) -> None:
        """
        Record one camera frame for which MediaPipe found no target hand.

        Schema is kept identical to the pre-refactor Gate 3A server.
        """

        record = {
            "seq": seq,
            "timestamp": float(timestamp),
            "capture_monotonic_s": float(capture_monotonic_s),
            "tracking_valid": False,
            "joint_pos": None,
            "keypoint_2d": None,
            "reference_vectors_raw": None,
            "reference_vectors_aligned": None,
            "reference_vector_lengths": None,
            "positions_rad": None,
            "source": "mediapipe_dex_retargeting",
        }

        self._write(record)

    def write_valid(
        self,
        *,
        packet: dict,
        capture_monotonic_s: float,
        joint_pos,
        keypoint_2d,
        reference_vectors_raw: np.ndarray,
        reference_vectors_aligned: np.ndarray,
        reference_vector_lengths: np.ndarray,
    ) -> None:
        """
        Record one valid perception/retargeting frame.

        ``packet`` is copied first so the network packet and JSONL retain
        the same joint order, qpos, timestamp, source, and retarget time as
        in the original server.
        """

        if self._file is None:
            return

        keypoint_2d_array = (
            mediapipe_landmark_list_to_array(
                keypoint_2d
            )
        )

        record = dict(packet)

        record["capture_monotonic_s"] = float(
            capture_monotonic_s
        )

        record["joint_pos"] = (
            np.asarray(
                joint_pos,
                dtype=float,
            ).tolist()
        )

        record["keypoint_2d"] = (
            None
            if keypoint_2d_array is None
            else keypoint_2d_array.astype(float).tolist()
        )

        record["reference_vectors_raw"] = (
            np.asarray(reference_vectors_raw)
            .astype(float)
            .tolist()
        )

        record["reference_vectors_aligned"] = (
            np.asarray(reference_vectors_aligned)
            .astype(float)
            .tolist()
        )

        record["reference_vector_lengths"] = (
            np.asarray(reference_vector_lengths)
            .astype(float)
            .tolist()
        )

        self._write(record)

    def close(self) -> None:
        """Close the JSONL file if recording is enabled."""

        if self._file is not None:
            self._file.close()
            self._file = None
