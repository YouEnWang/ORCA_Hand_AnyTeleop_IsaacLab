"""Controlled jitter injection for denoising benchmarks."""

from __future__ import annotations

import numpy as np


def inject_jitter(
    values: np.ndarray,
    time_values: np.ndarray,
    amplitude_rad: float,
    frequency_hz: float,
    gaussian_std_rad: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    time_values = np.asarray(time_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(values.shape[1],))
    sinusoid = amplitude_rad * np.sin(
        2.0 * np.pi * frequency_hz * time_values[:, None] + phase[None, :]
    )
    gaussian = rng.normal(0.0, gaussian_std_rad, size=values.shape)
    noise = sinusoid + gaussian
    return values + noise, noise

