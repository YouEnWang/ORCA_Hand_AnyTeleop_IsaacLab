"""Dataset loading and saving helpers for ORCA teleoperation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


CORE_TRAJECTORY_KEYS = (
    "time",
    "joint_names",
    "q_desired",
    "q_command",
    "q_actual",
)


def ensure_parent(path: Path) -> None:
    path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.expanduser().open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            file.write("\n")


def save_npz(path: Path, dataset: dict[str, Any]) -> None:
    ensure_parent(path)
    arrays = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, list) and key == "joint_names":
            arrays[key] = np.asarray(value, dtype=str)
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path.expanduser(), **arrays)


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path.expanduser(), allow_pickle=False) as data:
        dataset = {key: data[key] for key in data.files}
    if "joint_names" in dataset:
        dataset["joint_names"] = [str(name) for name in dataset["joint_names"].tolist()]
    return dataset


def load_dataset(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".jsonl":
        return jsonl_to_dataset(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def jsonl_to_dataset(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    if not records:
        raise RuntimeError(f"No records found in {path}")

    joint_names = None
    rows: dict[str, list[Any]] = {
        "time": [],
        "seq": [],
        "tracking_valid": [],
    }
    vector_keys = ("q_desired", "q_command", "q_actual", "positions_rad")

    for record in records:
        if not any(key in record for key in vector_keys):
            continue

        if joint_names is None and "joint_names" in record:
            joint_names = [str(name) for name in record["joint_names"]]

        timestamp = record.get("t_sim", record.get("timestamp", record.get("time")))
        rows["time"].append(float(timestamp))
        rows["seq"].append(int(record.get("seq", -1)))
        rows["tracking_valid"].append(bool(record.get("tracking_valid", True)))

        for key in vector_keys:
            if key in record:
                rows.setdefault(key, []).append(record[key])

    dataset: dict[str, Any] = {
        "time": np.asarray(rows["time"], dtype=np.float64),
        "seq": np.asarray(rows["seq"], dtype=np.int64),
        "tracking_valid": np.asarray(rows["tracking_valid"], dtype=bool),
        "joint_names": joint_names or [],
    }

    if len(dataset["time"]) == 0:
        raise RuntimeError(f"No trajectory-bearing records found in {path}")

    for key in vector_keys:
        values = rows.get(key)
        if values:
            dataset[key] = np.asarray(values, dtype=np.float64)

    if "positions_rad" in dataset and "q_desired" not in dataset:
        dataset["q_desired"] = dataset["positions_rad"]

    return dataset


def require_trajectory(dataset: dict[str, Any], key: str) -> np.ndarray:
    if key not in dataset:
        raise KeyError(f"Dataset is missing required trajectory key: {key}")
    value = np.asarray(dataset[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must have shape (frames, joints), got {value.shape}")
    return value


def normalize_time(time_values: np.ndarray) -> np.ndarray:
    values = np.asarray(time_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"time must be 1-D, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("time is empty")
    return values - values[0]
