"""Purification filters for ORCA teleoperation demonstrations."""

from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    values = np.asarray(values, dtype=np.float64)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    output = np.empty_like(values)
    for joint_index in range(values.shape[1]):
        output[:, joint_index] = np.convolve(padded[:, joint_index], kernel, mode="valid")
    return output


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import medfilt

        return medfilt(values, kernel_size=(window, 1))
    except ImportError:
        pad = window // 2
        values = np.asarray(values, dtype=np.float64)
        padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
        output = np.empty_like(values)
        for frame in range(values.shape[0]):
            output[frame] = np.median(padded[frame : frame + window], axis=0)
        return output


def savitzky_golay(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window <= 2:
        return np.asarray(values, dtype=np.float64).copy()
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError("Savitzky-Golay filtering requires scipy") from exc
    window = min(window, values.shape[0] - (1 - values.shape[0] % 2))
    if window <= polyorder:
        window = polyorder + 2 + ((polyorder + 2) % 2 == 0)
    return savgol_filter(values, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def butterworth_lowpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:
        raise RuntimeError("Butterworth filtering requires scipy") from exc
    nyquist = 0.5 * sample_rate_hz
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be in (0, {nyquist}), got {cutoff_hz}")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, values, axis=0)


def clamp_velocity(values: np.ndarray, time_values: np.ndarray, max_speed_rad_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    time_values = np.asarray(time_values, dtype=np.float64)
    output = values.copy()
    for frame in range(1, values.shape[0]):
        dt = max(float(time_values[frame] - time_values[frame - 1]), 1e-9)
        max_delta = max_speed_rad_s * dt
        delta = np.clip(values[frame] - output[frame - 1], -max_delta, max_delta)
        output[frame] = output[frame - 1] + delta
    return output


def suppress_outliers(values: np.ndarray, z_threshold: float = 6.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = values.copy()
    delta = np.diff(values, axis=0, prepend=values[:1])
    median = np.median(delta, axis=0, keepdims=True)
    mad = np.median(np.abs(delta - median), axis=0, keepdims=True)
    robust_sigma = 1.4826 * np.maximum(mad, 1e-9)
    outliers = np.abs(delta - median) > z_threshold * robust_sigma
    for frame in range(1, values.shape[0]):
        mask = outliers[frame]
        output[frame, mask] = output[frame - 1, mask]
    return output


def apply_filter(
    values: np.ndarray,
    time_values: np.ndarray,
    method: str,
    **kwargs: float | int | str,
) -> np.ndarray:
    method = method.lower().replace("-", "_")
    if method == "none":
        return np.asarray(values, dtype=np.float64).copy()
    if method == "moving_average":
        return moving_average(values, int(kwargs.get("window", 5)))
    if method == "median":
        return median_filter(values, int(kwargs.get("window", 5)))
    if method == "savgol":
        return savitzky_golay(values, int(kwargs.get("window", 9)), int(kwargs.get("polyorder", 3)))
    if method == "butterworth":
        sample_rate = 1.0 / np.median(np.diff(time_values))
        return butterworth_lowpass(
            values,
            sample_rate_hz=float(sample_rate),
            cutoff_hz=float(kwargs.get("cutoff_hz", 6.0)),
            order=int(kwargs.get("order", 2)),
        )
    if method == "outlier_then_butterworth":
        clean = suppress_outliers(values, float(kwargs.get("z_threshold", 6.0)))
        sample_rate = 1.0 / np.median(np.diff(time_values))
        return butterworth_lowpass(
            clean,
            sample_rate_hz=float(sample_rate),
            cutoff_hz=float(kwargs.get("cutoff_hz", 6.0)),
            order=int(kwargs.get("order", 2)),
        )
    if method == "velocity_clamp":
        return clamp_velocity(values, time_values, float(kwargs.get("max_speed_rad_s", 2.0)))
    raise ValueError(f"Unknown purification method: {method}")

