#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone 1-DoF wrist retargeter for ORCA Hand v2.

Design goal
-----------
Keep wrist control separate from dex-retargeting VectorOptimizer:

    MediaPipe / dex-retargeting SingleHandDetector
        ├─ local finger pose -> VectorOptimizer -> 16 finger DoF
        └─ mediapipe_wrist_rot -> this module -> 1 wrist DoF

The returned wrist command is intended to be merged into qpos AFTER
VectorOptimizer has solved the 16 finger joints with the wrist fixed at zero.

Coordinate convention
---------------------
SingleHandDetector returns:
    mediapipe_wrist_rot : 3x3 wrist frame estimated by MediaPipe
and internally uses:
    joint_pos = raw_wrist_centered @ mediapipe_wrist_rot @ operator2mano

For wrist motion we:
1. Calibrate a neutral MediaPipe wrist frame C0.
2. Compute relative human wrist rotation:
       R_h = O.T @ C0.T @ C @ O
   where O = operator2mano.
3. Convert the relative rotation into the calibrated ORCA vector frame:
       R_orca = A @ R_h @ A.T
   where A is the existing Human->ORCA alignment rotation.
4. Extract the twist component around the ORCA wrist joint axis.
5. Apply sign/gain/deadband, safe limits, EMA smoothing, and a rate limit.

No Isaac Sim dependency is used here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


_EPS = 1e-10


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """
    Project a nearly-rotation matrix to the closest proper rotation matrix.
    """
    m = np.asarray(matrix, dtype=np.float64)

    if m.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {m.shape}.")

    if not np.all(np.isfinite(m)):
        raise ValueError("Rotation matrix contains NaN/Inf.")

    u, _, vt = np.linalg.svd(m)
    r = u @ vt

    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt

    return r.astype(np.float64)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].
    """
    r = project_to_so3(rotation)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s

    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(
            1.0 + r[0, 0] - r[1, 1] - r[2, 2]
        ) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s

    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(
            1.0 + r[1, 1] - r[0, 0] - r[2, 2]
        ) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s

    else:
        s = math.sqrt(
            1.0 + r[2, 2] - r[0, 0] - r[1, 1]
        ) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    q = np.asarray(
        [w, x, y, z],
        dtype=np.float64,
    )

    norm = float(np.linalg.norm(q))

    if norm < _EPS:
        raise ValueError("Degenerate quaternion from rotation matrix.")

    q /= norm

    # q and -q represent the same rotation.
    # Keep w >= 0 to obtain a stable short-angle representation near neutral.
    if q[0] < 0.0:
        q *= -1.0

    return q


def twist_angle_about_axis(
    rotation: np.ndarray,
    axis: np.ndarray,
) -> float:
    """
    Extract the signed twist angle of `rotation` around a specified unit axis.

    This is a swing-twist decomposition. It is preferable to selecting one
    Euler angle because Euler angles introduce order-dependent cross-coupling.
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)

    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < _EPS:
        raise ValueError("Twist axis must be non-zero.")

    axis = axis / axis_norm

    q = rotation_matrix_to_quaternion(rotation)

    w = float(q[0])
    v = q[1:4]

    # Project quaternion vector part onto the desired twist axis.
    projected_v = axis * float(np.dot(v, axis))

    twist_norm = math.sqrt(
        w * w
        + float(np.dot(projected_v, projected_v))
    )

    # 180-degree swing exactly orthogonal to the twist axis is singular.
    # Around our neutral operating region it should not occur.
    if twist_norm < _EPS:
        return 0.0

    twist_w = w / twist_norm
    twist_v = projected_v / twist_norm

    signed_sin_half = float(
        np.dot(
            twist_v,
            axis,
        )
    )

    angle = 2.0 * math.atan2(
        signed_sin_half,
        twist_w,
    )

    return wrap_to_pi(angle)


@dataclass
class WristDiagnostics:
    ready: bool
    neutral_count: int
    neutral_frames: int

    raw_twist_rad: float = 0.0
    mapped_rad: float = 0.0
    filtered_rad: float = 0.0
    command_rad: float = 0.0

    clamped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": bool(self.ready),
            "neutral_count": int(self.neutral_count),
            "neutral_frames": int(self.neutral_frames),
            "raw_twist_rad": float(self.raw_twist_rad),
            "mapped_rad": float(self.mapped_rad),
            "filtered_rad": float(self.filtered_rad),
            "command_rad": float(self.command_rad),
            "clamped": bool(self.clamped),
        }


class OrcaWristRetargeter:
    """
    MediaPipe wrist orientation -> ORCA Hand v2 single wrist joint.
    """

    def __init__(
        self,
        *,
        joint_name: str,
        human_to_orca_rotation: np.ndarray,
        joint_axis_orca: np.ndarray,
        neutral_frames: int,
        sign: float,
        gain: float,
        offset_rad: float,
        deadband_rad: float,
        filter_alpha: float,
        max_speed_rad_s: float,
        safe_lower_rad: float,
        safe_upper_rad: float,
    ):
        self.joint_name = str(joint_name)

        self.alignment_rotation = project_to_so3(
            human_to_orca_rotation
        )

        axis = np.asarray(
            joint_axis_orca,
            dtype=np.float64,
        ).reshape(3)

        axis_norm = float(
            np.linalg.norm(axis)
        )

        if axis_norm < _EPS:
            raise ValueError(
                "joint_axis_orca must be non-zero."
            )

        self.joint_axis_orca = (
            axis
            / axis_norm
        )

        self.neutral_frames = int(
            neutral_frames
        )

        if self.neutral_frames < 1:
            raise ValueError(
                "neutral_frames must be >= 1."
            )

        self.sign = float(sign)

        if abs(self.sign) < _EPS:
            raise ValueError(
                "sign cannot be zero."
            )

        self.gain = float(gain)

        if self.gain <= 0.0:
            raise ValueError(
                "gain must be positive."
            )

        self.offset_rad = float(
            offset_rad
        )

        self.deadband_rad = float(
            deadband_rad
        )

        if self.deadband_rad < 0.0:
            raise ValueError(
                "deadband_rad cannot be negative."
            )

        self.filter_alpha = float(
            filter_alpha
        )

        if not (
            0.0
            < self.filter_alpha
            <= 1.0
        ):
            raise ValueError(
                "filter_alpha must be in (0, 1]."
            )

        self.max_speed_rad_s = float(
            max_speed_rad_s
        )

        if self.max_speed_rad_s <= 0.0:
            raise ValueError(
                "max_speed_rad_s must be positive."
            )

        self.safe_lower_rad = float(
            safe_lower_rad
        )

        self.safe_upper_rad = float(
            safe_upper_rad
        )

        if (
            self.safe_lower_rad
            >= self.safe_upper_rad
        ):
            raise ValueError(
                "safe_lower_rad must be < safe_upper_rad."
            )

        if not (
            self.safe_lower_rad
            <= self.offset_rad
            <= self.safe_upper_rad
        ):
            raise ValueError(
                "offset_rad must lie inside the safe range."
            )

        self.reset_neutral()

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | str,
        *,
        human_to_orca_rotation: np.ndarray,
    ) -> "OrcaWristRetargeter":

        path = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        required = [
            "wrist_joint_name",
            "joint_axis_orca",
            "neutral_frames",
            "sign",
            "gain",
            "offset_rad",
            "deadband_rad",
            "filter_alpha",
            "max_speed_rad_s",
            "safe_lower_rad",
            "safe_upper_rad",
        ]

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:
            raise KeyError(
                "Wrist config missing fields: "
                + ", ".join(missing)
            )

        return cls(
            joint_name=data["wrist_joint_name"],
            human_to_orca_rotation=human_to_orca_rotation,
            joint_axis_orca=data["joint_axis_orca"],
            neutral_frames=data["neutral_frames"],
            sign=data["sign"],
            gain=data["gain"],
            offset_rad=data["offset_rad"],
            deadband_rad=data["deadband_rad"],
            filter_alpha=data["filter_alpha"],
            max_speed_rad_s=data["max_speed_rad_s"],
            safe_lower_rad=data["safe_lower_rad"],
            safe_upper_rad=data["safe_upper_rad"],
        )

    def reset_neutral(self) -> None:
        """Start a new neutral-pose calibration."""
        self._neutral_samples: list[np.ndarray] = []
        self._neutral_rotation: np.ndarray | None = None

        self._filtered_command = float(
            self.offset_rad
        )

        self._rate_limited_command = float(
            self.offset_rad
        )

        self._last_timestamp: float | None = None

        self.last_diagnostics = WristDiagnostics(
            ready=False,
            neutral_count=0,
            neutral_frames=self.neutral_frames,
            command_rad=float(
                self.offset_rad
            ),
        )

    @property
    def ready(self) -> bool:
        return (
            self._neutral_rotation
            is not None
        )

    def _finish_neutral_calibration(
        self,
        timestamp: float,
    ) -> None:

        mean_basis = np.mean(
            np.stack(
                self._neutral_samples,
                axis=0,
            ),
            axis=0,
        )

        self._neutral_rotation = (
            project_to_so3(
                mean_basis
            )
        )

        self._filtered_command = float(
            self.offset_rad
        )

        self._rate_limited_command = float(
            self.offset_rad
        )

        self._last_timestamp = float(
            timestamp
        )

    def update(
        self,
        mediapipe_wrist_rot: np.ndarray,
        operator2mano: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> WristDiagnostics:

        now = (
            time.monotonic()
            if timestamp is None
            else float(timestamp)
        )

        current_basis = project_to_so3(
            mediapipe_wrist_rot
        )

        operator2mano = np.asarray(
            operator2mano,
            dtype=np.float64,
        )

        if operator2mano.shape != (3, 3):
            raise ValueError(
                "operator2mano must have shape (3, 3)."
            )

        if not np.all(
            np.isfinite(operator2mano)
        ):
            raise ValueError(
                "operator2mano contains NaN/Inf."
            )

        ortho_error = float(
            np.max(
                np.abs(
                    operator2mano.T
                    @ operator2mano
                    -
                    np.eye(3)
                )
            )
        )

        if ortho_error > 1e-4:
            raise ValueError(
                "operator2mano is not orthogonal; "
                f"error={ortho_error:.3e}"
            )

        # ----------------------------------------------------------
        # Neutral calibration
        # ----------------------------------------------------------

        if not self.ready:

            self._neutral_samples.append(
                current_basis.copy()
            )

            if (
                len(self._neutral_samples)
                >= self.neutral_frames
            ):
                self._finish_neutral_calibration(
                    now
                )

            diagnostics = WristDiagnostics(
                ready=self.ready,
                neutral_count=min(
                    len(
                        self._neutral_samples
                    ),
                    self.neutral_frames,
                ),
                neutral_frames=self.neutral_frames,
                command_rad=float(
                    self.offset_rad
                ),
            )

            self.last_diagnostics = diagnostics
            return diagnostics

        # ----------------------------------------------------------
        # Relative human wrist orientation
        #
        # SingleHandDetector local/MANO coordinates:
        #   row-vector transform = C @ O
        #
        # Relative orientation in the same local convention:
        #   R_h = O.T @ C0.T @ C @ O
        # ----------------------------------------------------------

        assert (
            self._neutral_rotation
            is not None
        )

        relative_mediapipe = (
            self._neutral_rotation.T
            @ current_basis
        )

        relative_human = (
            operator2mano.T
            @ relative_mediapipe
            @ operator2mano
        )

        relative_human = project_to_so3(
            relative_human
        )

        # ----------------------------------------------------------
        # Human local frame -> ORCA frame.
        #
        # Existing vector calibration:
        #   v_orca = A @ v_human
        #
        # Therefore a rotation transforms by conjugation:
        #   R_orca = A @ R_h @ A.T
        # ----------------------------------------------------------

        relative_orca = (
            self.alignment_rotation
            @ relative_human
            @ self.alignment_rotation.T
        )

        relative_orca = project_to_so3(
            relative_orca
        )

        raw_twist = twist_angle_about_axis(
            relative_orca,
            self.joint_axis_orca,
        )

        # ----------------------------------------------------------
        # Scalar human -> robot mapping
        # ----------------------------------------------------------

        motion = (
            self.sign
            * self.gain
            * raw_twist
        )

        if abs(motion) < self.deadband_rad:
            motion = 0.0

        mapped = (
            self.offset_rad
            + motion
        )

        safe_target = float(
            np.clip(
                mapped,
                self.safe_lower_rad,
                self.safe_upper_rad,
            )
        )

        clamped = (
            abs(
                safe_target
                -
                mapped
            )
            > 1e-9
        )

        # ----------------------------------------------------------
        # EMA filtering
        # ----------------------------------------------------------

        filtered = (
            self.filter_alpha
            * safe_target
            +
            (
                1.0
                - self.filter_alpha
            )
            * self._filtered_command
        )

        self._filtered_command = float(
            filtered
        )

        # ----------------------------------------------------------
        # Command rate limit
        # ----------------------------------------------------------

        if self._last_timestamp is None:
            command = float(
                filtered
            )

        else:
            dt = max(
                now
                -
                self._last_timestamp,
                0.0,
            )

            max_delta = (
                self.max_speed_rad_s
                * dt
            )

            delta = float(
                filtered
                -
                self._rate_limited_command
            )

            delta = float(
                np.clip(
                    delta,
                    -max_delta,
                    +max_delta,
                )
            )

            command = float(
                self._rate_limited_command
                + delta
            )

        command = float(
            np.clip(
                command,
                self.safe_lower_rad,
                self.safe_upper_rad,
            )
        )

        self._rate_limited_command = (
            command
        )

        self._last_timestamp = now

        diagnostics = WristDiagnostics(
            ready=True,
            neutral_count=self.neutral_frames,
            neutral_frames=self.neutral_frames,
            raw_twist_rad=float(
                raw_twist
            ),
            mapped_rad=float(
                mapped
            ),
            filtered_rad=float(
                filtered
            ),
            command_rad=float(
                command
            ),
            clamped=bool(
                clamped
            ),
        )

        self.last_diagnostics = diagnostics

        return diagnostics
