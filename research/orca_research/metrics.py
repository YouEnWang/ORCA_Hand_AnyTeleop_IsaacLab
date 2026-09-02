"""Signal quality metrics for teleoperation demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    high_frequency_cutoff_hz: float = 6.0
    eps: float = 1e-12


def estimate_sample_rate_hz(time_values: np.ndarray) -> float:
    time_values = np.asarray(time_values, dtype=np.float64)
    if len(time_values) < 2:
        raise ValueError("Need at least two timestamps to estimate sample rate")
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        raise ValueError("Timestamps are not strictly increasing")
    return float(1.0 / np.median(dt))


def derivative(values: np.ndarray, time_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    time_values = np.asarray(time_values, dtype=np.float64)
    if values.shape[0] != time_values.shape[0]:
        raise ValueError("values and time length mismatch")
    return np.gradient(values, time_values, axis=0, edge_order=1)


def rms(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.nanmean(values * values, axis=axis))


def total_variation(values: np.ndarray) -> np.ndarray:
    return np.nansum(np.abs(np.diff(values, axis=0)), axis=0)


def high_frequency_energy_ratio(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    eps: float = 1e-12,
) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - np.nanmean(centered, axis=0, keepdims=True)
    centered = np.nan_to_num(centered)
    spectrum = np.fft.rfft(centered, axis=0)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(centered.shape[0], d=1.0 / sample_rate_hz)
    high = power[freqs >= cutoff_hz].sum(axis=0)
    total = power.sum(axis=0)
    return high / np.maximum(total, eps)


def spectral_entropy(values: np.ndarray, sample_rate_hz: float, eps: float = 1e-12) -> np.ndarray:
    del sample_rate_hz
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - np.nanmean(centered, axis=0, keepdims=True)
    spectrum = np.fft.rfft(np.nan_to_num(centered), axis=0)
    power = np.abs(spectrum) ** 2
    probability = power / np.maximum(power.sum(axis=0, keepdims=True), eps)
    entropy = -np.sum(probability * np.log2(np.maximum(probability, eps)), axis=0)
    max_entropy = math.log2(max(probability.shape[0], 2))
    return entropy / max_entropy


def tracking_error_metrics(command: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    if command.shape != actual.shape:
        raise ValueError(f"command and actual shapes differ: {command.shape} vs {actual.shape}")
    error = command - actual
    return {
        "tracking_rmse": rms(error, axis=0),
        "tracking_mae": np.nanmean(np.abs(error), axis=0),
        "tracking_max_abs": np.nanmax(np.abs(error), axis=0),
    }


def compute_quality_metrics(
    time_values: np.ndarray,
    q: np.ndarray,
    config: QualityConfig | None = None,
) -> dict[str, np.ndarray | float]:
    cfg = config or QualityConfig()
    t = np.asarray(time_values, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sample_rate_hz = estimate_sample_rate_hz(t)
    velocity = derivative(q, t)
    acceleration = derivative(velocity, t)
    jerk = derivative(acceleration, t)
    return {
        "sample_rate_hz": sample_rate_hz,
        "position_rms": rms(q, axis=0),
        "velocity_rms": rms(velocity, axis=0),
        "acceleration_rms": rms(acceleration, axis=0),
        "jerk_rms": rms(jerk, axis=0),
        "total_variation": total_variation(q),
        "high_frequency_energy_ratio": high_frequency_energy_ratio(
            q,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=cfg.high_frequency_cutoff_hz,
            eps=cfg.eps,
        ),
        "spectral_entropy": spectral_entropy(q, sample_rate_hz=sample_rate_hz, eps=cfg.eps),
    }


def summarize_joint_metrics(
    joint_names: list[str],
    metrics: dict[str, np.ndarray | float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    array_keys = [key for key, value in metrics.items() if isinstance(value, np.ndarray)]
    if not array_keys:
        return rows
    joint_count = len(np.asarray(metrics[array_keys[0]]))
    names = joint_names or [f"joint_{idx:02d}" for idx in range(joint_count)]
    for idx in range(joint_count):
        row: dict[str, float | str] = {"joint": names[idx] if idx < len(names) else f"joint_{idx:02d}"}
        for key in array_keys:
            row[key] = float(np.asarray(metrics[key])[idx])
        rows.append(row)
    return rows

