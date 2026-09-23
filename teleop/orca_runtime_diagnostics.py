#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Console diagnostics and runtime statistics for ORCA AnyTeleop.

Purpose
-------
The original server accumulated several hundred lines of diagnostic
printing inside the real-time loop.  This module moves that presentation
logic out of the control path while preserving the same measurements and
console semantics.

It has two responsibilities:

1. Static/startup diagnostics
   - ORCA initial pose and joint limits.
   - Human reference-vector landmark mapping.
   - Selected retarget/alignment configuration.
   - Human -> ORCA alignment matrix.
   - ORCA qpos joint order.

2. Runtime diagnostics
   - Per-second capture/tracking/retarget/network counters.
   - Raw human vector values, lengths, and frame-to-frame changes.
   - Aligned-vector values and lengths.
   - ORCA qpos values and joint-limit saturation markers.

This module does not change camera frames, vectors, qpos, UDP packets, or
JSONL data.  It is intentionally observation-only so refactoring it out
of the main loop does not change the experiment algorithm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


MEDIAPIPE_LANDMARK_NAMES = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_mcp",
    6: "index_pip",
    7: "index_dip",
    8: "index_tip",
    9: "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}


def print_initial_pose(
    *,
    joint_names: Sequence[str],
    initial_qpos: np.ndarray,
    joint_limits: np.ndarray,
) -> None:
    """Print the same ORCA initial-pose table used before refactoring."""

    print()
    print("=" * 80)
    print("ORCA retargeting initial pose")
    print("=" * 80)

    for i, name in enumerate(joint_names):
        print(
            f"[{i:02d}] "
            f"{name:<65} "
            f"q0={initial_qpos[i]:+.4f} "
            f"range="
            f"[{joint_limits[i, 0]:+.4f}, "
            f"{joint_limits[i, 1]:+.4f}]"
        )


def print_camera_configuration(cap) -> None:
    """Print the camera backend and the settings actually reported by OpenCV."""

    import cv2

    print()
    print("=" * 80)
    print("Camera runtime configuration")
    print("=" * 80)

    print(
        "Backend           :",
        (
            cap.getBackendName()
            if hasattr(cap, "getBackendName")
            else "unknown"
        ),
    )
    print(
        "Actual width      :",
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    )
    print(
        "Actual height     :",
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    print(
        "Reported FPS      :",
        cap.get(cv2.CAP_PROP_FPS),
    )


def print_human_vector_mapping(
    *,
    origin_indices: np.ndarray,
    task_indices: np.ndarray,
) -> None:
    """Print MediaPipe landmark pairs used to construct each human vector."""

    print()
    print("=" * 80)
    print("Human reference-vector mapping")
    print("=" * 80)

    for vector_index, (
        origin_index,
        task_index,
    ) in enumerate(zip(origin_indices, task_indices)):
        origin_index = int(origin_index)
        task_index = int(task_index)

        origin_name = MEDIAPIPE_LANDMARK_NAMES.get(
            origin_index,
            f"landmark_{origin_index}",
        )
        task_name = MEDIAPIPE_LANDMARK_NAMES.get(
            task_index,
            f"landmark_{task_index}",
        )

        print(
            f"[HV{vector_index:02d}] "
            f"MP{origin_index:02d} "
            f"{origin_name:<12} "
            f"-> "
            f"MP{task_index:02d} "
            f"{task_name}"
        )


def print_server_configuration(
    *,
    camera_source,
    hand_type: str,
    selfie: bool,
    retarget_config: Path,
    alignment_config: Optional[Path],
    dry_run: bool,
    retargeting_type,
    joint_names: Sequence[str],
    optimized_dof: int,
    fixed_joint_names: Sequence[str],
    host: str,
    port: int,
) -> None:
    """Print the selected Gate 3 server/runtime configuration."""

    print()
    print("=" * 80)
    print("AnyTeleop ORCA Hand Server")
    print("=" * 80)

    print("Camera            :", camera_source)
    print("Hand type         :", hand_type)
    print("Selfie            :", selfie)
    print(
        "Retarget config   :",
        retarget_config.expanduser().resolve(),
    )
    print(
        "Alignment config  :",
        (
            alignment_config.expanduser().resolve()
            if alignment_config is not None
            else "None"
        ),
    )
    print("Dry run           :", dry_run)
    print("UDP enabled       :", not dry_run)
    print("Retargeting       :", retargeting_type)
    print("Robot DOF         :", len(joint_names))
    print("Optimized DOF     :", optimized_dof)
    print("Fixed joints      :", list(fixed_joint_names))
    print("UDP destination   :", f"{host}:{port}")


def print_alignment_configuration(
    *,
    alignment_enabled: bool,
    alignment_config: Optional[Path],
    alignment_scale: float,
    alignment_det: float,
    alignment_orth_error: float,
    alignment_rotation: np.ndarray,
) -> None:
    """Print Human -> ORCA calibration information."""

    print()
    print("=" * 80)
    print("Human -> ORCA vector alignment")
    print("=" * 80)

    print("Enabled           :", alignment_enabled)

    if alignment_enabled:
        print(
            "Alignment config  :",
            alignment_config.resolve(),
        )

    print("Scale             :", f"{alignment_scale:.8f}")
    print("det(R)            :", f"{alignment_det:.8f}")
    print(
        "Orthogonality err :",
        f"{alignment_orth_error:.8e}",
    )
    print("Rotation R:")
    print(alignment_rotation)
    print()


def print_robot_joint_order(
    joint_names: Sequence[str],
) -> None:
    """Print qpos array index -> URDF joint name mapping."""

    print()
    print("ORCA robot joint order:")

    for i, name in enumerate(joint_names):
        print(f"  [{i:02d}] {name}")


@dataclass
class RuntimeDiagnostics:
    """
    Hold and print the server's one-second runtime diagnostics.

    The class centralizes only diagnostic state.  Control state used by
    the optimizer remains in the main server.  Counter reset behavior and
    qpos/vector delta calculations match the pre-refactor implementation.
    """

    origin_indices: np.ndarray
    task_indices: np.ndarray
    joint_names: Sequence[str]
    joint_limits: np.ndarray
    alignment_scale: float
    run_start_time: float

    captured_count: int = 0
    read_failure_count: int = 0
    valid_detection_count: int = 0
    invalid_detection_count: int = 0
    retarget_success_count: int = 0
    retarget_failure_count: int = 0
    sent_valid_count: int = 0
    sent_invalid_count: int = 0

    previous_qpos: Optional[np.ndarray] = None
    latest_qpos: Optional[np.ndarray] = None
    latest_q_delta_max: float = 0.0
    latest_retarget_ms: float = 0.0

    previous_reference_vectors: Optional[np.ndarray] = None
    latest_reference_vectors: Optional[np.ndarray] = None
    latest_reference_lengths: Optional[np.ndarray] = None
    latest_reference_delta_lengths: Optional[np.ndarray] = None
    latest_reference_delta_vectors: Optional[np.ndarray] = None

    latest_aligned_vectors: Optional[np.ndarray] = None
    latest_aligned_lengths: Optional[np.ndarray] = None

    status_timer: float = field(default_factory=time.monotonic)

    def on_capture_success(self) -> None:
        self.captured_count += 1

    def on_read_failure(self) -> None:
        self.read_failure_count += 1

    def on_detection_valid(self) -> None:
        self.valid_detection_count += 1

    def on_detection_invalid(self) -> None:
        self.invalid_detection_count += 1

    def on_retarget_success(self) -> None:
        self.retarget_success_count += 1

    def on_retarget_failure(self) -> None:
        self.retarget_failure_count += 1

    def on_sent_valid(self) -> None:
        self.sent_valid_count += 1

    def on_sent_invalid(self) -> None:
        self.sent_invalid_count += 1

    def update_reference_vectors(
        self,
        reference_vectors_raw: np.ndarray,
    ) -> None:
        """Update raw-vector length and frame-to-frame delta diagnostics."""

        latest = reference_vectors_raw.copy()
        lengths = np.linalg.norm(latest, axis=1)

        if self.previous_reference_vectors is None:
            delta_lengths = np.zeros_like(lengths)
            delta_vectors = np.zeros_like(lengths)
        else:
            previous_lengths = np.linalg.norm(
                self.previous_reference_vectors,
                axis=1,
            )
            delta_lengths = lengths - previous_lengths
            delta_vectors = np.linalg.norm(
                latest - self.previous_reference_vectors,
                axis=1,
            )

        self.latest_reference_vectors = latest
        self.latest_reference_lengths = lengths
        self.latest_reference_delta_lengths = delta_lengths
        self.latest_reference_delta_vectors = delta_vectors
        self.previous_reference_vectors = latest.copy()

    def update_aligned_vectors(
        self,
        reference_vectors_aligned: np.ndarray,
    ) -> None:
        """Update aligned-vector value/length diagnostics."""

        self.latest_aligned_vectors = (
            reference_vectors_aligned.copy()
        )
        self.latest_aligned_lengths = np.linalg.norm(
            self.latest_aligned_vectors,
            axis=1,
        )

    def update_qpos(
        self,
        qpos: np.ndarray,
        retarget_ms: float,
    ) -> None:
        """Update latest qpos and frame-to-frame maximum qpos change."""

        self.latest_retarget_ms = float(retarget_ms)
        self.latest_qpos = qpos.copy()

        if self.previous_qpos is None:
            self.latest_q_delta_max = 0.0
        else:
            self.latest_q_delta_max = float(
                np.max(
                    np.abs(qpos - self.previous_qpos)
                )
            )

        self.previous_qpos = qpos.copy()

    def _due(self, now: float) -> bool:
        return now - self.status_timer >= 1.0

    def _valid_ratio(self) -> float:
        return (
            100.0
            * self.valid_detection_count
            / max(self.captured_count, 1)
        )

    def _reset_one_second_counters(self, now: float) -> None:
        self.captured_count = 0
        self.read_failure_count = 0
        self.valid_detection_count = 0
        self.invalid_detection_count = 0
        self.retarget_success_count = 0
        self.retarget_failure_count = 0
        self.sent_valid_count = 0
        self.sent_invalid_count = 0
        self.status_timer = now

    def maybe_print_invalid(self) -> None:
        """
        Print the compact status line used while the current frame is invalid.

        This intentionally does not print stale vector/qpos detail blocks.
        """

        now = time.monotonic()
        if not self._due(now):
            return

        print(
            f"[SERVER] "
            f"camera={self.captured_count:3d}/s  "
            f"read_fail={self.read_failure_count:3d}  "
            f"valid={self.valid_detection_count:3d}  "
            f"invalid={self.invalid_detection_count:3d}  "
            f"ratio={self._valid_ratio():6.2f}%  "
            f"retarget_ok={self.retarget_success_count:3d}  "
            f"retarget_fail={self.retarget_failure_count:3d}  "
            f"sent_valid={self.sent_valid_count:3d}  "
            f"sent_invalid={self.sent_invalid_count:3d}"
        )

        self._reset_one_second_counters(now)

    def maybe_print_valid(self, seq: int) -> None:
        """Print the detailed once-per-second status used after a valid frame."""

        now = time.monotonic()
        if not self._due(now):
            return

        if self.latest_qpos is not None:
            q_min = float(np.min(self.latest_qpos))
            q_max = float(np.max(self.latest_qpos))
            q_text = f"[{q_min:+.3f}, {q_max:+.3f}]"
        else:
            q_text = "N/A"

        print(
            f"[SERVER] "
            f"camera={self.captured_count:3d}/s  "
            f"read_fail={self.read_failure_count:3d}  "
            f"valid={self.valid_detection_count:3d}  "
            f"invalid={self.invalid_detection_count:3d}  "
            f"ratio={self._valid_ratio():6.2f}%  "
            f"retarget_ok={self.retarget_success_count:3d}  "
            f"retarget_fail={self.retarget_failure_count:3d}  "
            f"sent={self.sent_valid_count:3d}  "
            f"retarget={self.latest_retarget_ms:6.2f}ms  "
            f"q={q_text}  "
            f"dq_max={self.latest_q_delta_max:.4f}  "
            f"seq={seq}"
        )

        self._print_human_vectors()
        self._print_aligned_vectors()
        self._print_qpos()

        self._reset_one_second_counters(now)

    def _print_human_vectors(self) -> None:
        if (
            self.latest_reference_vectors is None
            or self.latest_reference_lengths is None
        ):
            return

        elapsed_time = time.monotonic() - self.run_start_time
        print(f"[HVEC] t={elapsed_time:6.2f}s")

        for vector_index in range(
            len(self.latest_reference_vectors)
        ):
            origin_index = int(
                self.origin_indices[vector_index]
            )
            task_index = int(
                self.task_indices[vector_index]
            )

            origin_name = MEDIAPIPE_LANDMARK_NAMES.get(
                origin_index,
                f"landmark_{origin_index}",
            )
            task_name = MEDIAPIPE_LANDMARK_NAMES.get(
                task_index,
                f"landmark_{task_index}",
            )

            vx, vy, vz = [
                float(v)
                for v in self.latest_reference_vectors[
                    vector_index
                ]
            ]
            length = float(
                self.latest_reference_lengths[vector_index]
            )
            delta_length = float(
                self.latest_reference_delta_lengths[
                    vector_index
                ]
            )
            delta_vector = float(
                self.latest_reference_delta_vectors[
                    vector_index
                ]
            )

            print(
                f"  [HV{vector_index:02d}] "
                f"{origin_name:>10} "
                f"-> "
                f"{task_name:<12} "
                f"v=[{vx:+.5f}, {vy:+.5f}, {vz:+.5f}]  "
                f"L={length:.5f}  "
                f"dL={delta_length:+.6f}  "
                f"dV={delta_vector:.6f}"
            )

    def _print_aligned_vectors(self) -> None:
        if (
            self.latest_aligned_vectors is None
            or self.latest_aligned_lengths is None
        ):
            return

        elapsed_time = time.monotonic() - self.run_start_time
        print(
            f"[ALIGN] "
            f"t={elapsed_time:6.2f}s  "
            f"scale={self.alignment_scale:.6f}"
        )

        for vector_index in range(
            len(self.latest_aligned_vectors)
        ):
            task_index = int(
                self.task_indices[vector_index]
            )
            task_name = MEDIAPIPE_LANDMARK_NAMES.get(
                task_index,
                f"landmark_{task_index}",
            )

            vx, vy, vz = [
                float(v)
                for v in self.latest_aligned_vectors[
                    vector_index
                ]
            ]
            length = float(
                self.latest_aligned_lengths[vector_index]
            )

            print(
                f"  [AV{vector_index:02d}] "
                f"{task_name:<12} "
                f"v=[{vx:+.5f}, {vy:+.5f}, {vz:+.5f}]  "
                f"L={length:.5f}"
            )

    def _print_qpos(self) -> None:
        if self.latest_qpos is None:
            return

        elapsed_time = time.monotonic() - self.run_start_time
        print(f"[QPOS] t={elapsed_time:6.2f}s")

        limit_tolerance = 0.003

        for joint_index, (
            joint_name,
            joint_value,
        ) in enumerate(
            zip(
                self.joint_names,
                self.latest_qpos,
            )
        ):
            value = float(joint_value)
            lower = float(
                self.joint_limits[joint_index, 0]
            )
            upper = float(
                self.joint_limits[joint_index, 1]
            )
            joint_range = upper - lower

            if joint_range > 1e-8:
                normalized = (
                    (value - lower) / joint_range
                )
            else:
                normalized = 0.0

            if value <= lower + limit_tolerance:
                limit_state = "<<< LOWER_LIMIT"
            elif value >= upper - limit_tolerance:
                limit_state = "<<< UPPER_LIMIT"
            else:
                limit_state = ""

            print(
                f"  [{joint_index:02d}] "
                f"{joint_name:<65} "
                f"q={value:+.4f}  "
                f"norm={normalized:6.3f}  "
                f"range=[{lower:+.4f}, {upper:+.4f}]  "
                f"{limit_state}"
            )
