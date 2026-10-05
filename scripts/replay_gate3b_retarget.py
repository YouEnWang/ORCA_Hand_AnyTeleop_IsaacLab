#!/usr/bin/env python3

"""
Gate 3B offline retargeting replay.

Purpose
-------
Replay the exact aligned human task-space vectors recorded during Gate 3A
through dex-retargeting without using the camera, UDP, or Isaac Lab.

For every valid frame this script records:

    1. q_optimizer_target
       Raw optimized target joints produced by VectorOptimizer.
       This is BEFORE the output LPFilter.

    2. q_robot_prefilter
       Full robot DoF vector after inserting fixed joints and applying the
       kinematic/mimic adaptor, but BEFORE LPFilter.

    3. q_robot_output
       Full robot DoF vector returned by SeqRetargeting.
       For alpha=0.2 this is the canonical filtered command.
       For alpha=1.0 this is an exact LPF bypass.

Important
---------
Each invocation creates a fresh SeqRetargeting object.

Do NOT reuse SeqRetargeting.reset() between trials because reset() resets
last_qpos but does not reset the LPFilter state in this dex-retargeting
version.

Invalid tracking frames do NOT update either the optimizer or LPFilter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate 3B offline ORCA retargeting replay"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Gate 3A JSONL input file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Gate 3B replay JSONL output file",
    )

    parser.add_argument(
        "--config",
        default=(
            "config/retargeting/"
            "orca_v2_right_vector_virtual_tip.yml"
        ),
        help="Canonical ORCA retargeting YAML",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help=(
            "dex-retargeting LPFilter alpha. "
            "Use 0.2 for canonical Gate 3A behavior and "
            "1.0 for exact LPF bypass."
        ),
    )

    parser.add_argument(
        "--fixed-value",
        type=float,
        default=0.0,
        help=(
            "Value assigned to every fixed robot DoF. "
            "For the current ORCA setup this is the fixed wrist at 0 rad."
        ),
    )

    return parser.parse_args()


def to_list(x):
    return np.asarray(x).tolist()


def build_retargeter(config_path: Path, alpha: float):
    """
    Build a completely fresh SeqRetargeting object.

    Using override avoids modifying the canonical YAML on disk.
    """

    cfg = RetargetingConfig.load_from_file(
        str(config_path),
        override={
            "low_pass_alpha": float(alpha),
        },
    )

    return cfg.build()


def reconstruct_prefilter_qpos(retargeter, fixed_qpos):
    """
    Reproduce the robot_qpos immediately before LPFilter.

    Mirrors SeqRetargeting.retarget():

        robot_qpos[fixed]  = fixed_qpos
        robot_qpos[target] = last_qpos

        if adaptor:
            robot_qpos = adaptor.forward_qpos(robot_qpos)
    """

    optimizer = retargeter.optimizer

    robot_qpos = np.zeros(
        optimizer.robot.dof,
        dtype=np.float64,
    )

    robot_qpos[optimizer.idx_pin2fixed] = fixed_qpos

    # SeqRetargeting.last_qpos is the raw optimizer result.
    robot_qpos[optimizer.idx_pin2target] = (
        np.asarray(retargeter.last_qpos, dtype=np.float64)
    )

    if optimizer.adaptor is not None:
        robot_qpos = optimizer.adaptor.forward_qpos(
            robot_qpos
        )

    return np.asarray(robot_qpos, dtype=np.float64)


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(args.config).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(
            "--alpha must be in [0, 1]"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # IMPORTANT:
    # One fresh retargeter for one complete trial.
    retargeter = build_retargeter(
        config_path,
        args.alpha,
    )

    optimizer = retargeter.optimizer

    robot_joint_names = list(
        optimizer.robot.dof_joint_names
    )

    target_joint_names = list(
        optimizer.target_joint_names
    )

    fixed_joint_names = [
        robot_joint_names[i]
        for i in optimizer.idx_pin2fixed
    ]

    n_fixed = len(optimizer.idx_pin2fixed)

    fixed_qpos = np.full(
        n_fixed,
        float(args.fixed_value),
        dtype=np.float32,
    )
    
    # ============================================================
    # Gate 3A-compatible initialization
    # ============================================================
    #
    # Gate 3A live server does NOT use SeqRetargeting's default
    # joint-limit-midpoint initialization.
    #
    # Instead:
    #
    #   1. Start every robot DoF from 0 rad.
    #   2. Clamp 0 rad into each joint's legal range.
    #   3. Set fixed joints (currently the wrist) to the same
    #      fixed value used during replay.
    #   4. Call set_qpos() BEFORE processing the first frame.
    #
    # This is required for replay fidelity because VectorOptimizer
    # is stateful and uses the previous qpos as the next optimization
    # initial condition.
    # ============================================================

    robot = optimizer.robot

    robot_joint_limits = np.asarray(
        robot.joint_limits,
        dtype=np.float32,
    )

    initial_qpos = np.zeros(
        robot.dof,
        dtype=np.float32,
    )

    # Use zero whenever zero is legal; otherwise use the closest
    # valid boundary value.
    initial_qpos = np.clip(
        initial_qpos,
        robot_joint_limits[:, 0],
        robot_joint_limits[:, 1],
    )

    # Match the fixed DoF used during Gate 3A / replay.
    if n_fixed > 0:
        initial_qpos[
            optimizer.idx_pin2fixed
        ] = np.clip(
            fixed_qpos,
            robot_joint_limits[
                optimizer.idx_pin2fixed, 0
            ],
            robot_joint_limits[
                optimizer.idx_pin2fixed, 1
            ],
        )

    # Override SeqRetargeting's default midpoint initialization
    # exactly as done by the Gate 3A live server.
    retargeter.set_qpos(
        initial_qpos
    )

    target_limits = np.asarray(
        retargeter.joint_limits,
        dtype=np.float64,
    )

    print("=" * 72)
    print("[Gate 3B replay]")
    print("Input       :", input_path)
    print("Output      :", output_path)
    print("Config      :", config_path)
    print("LPF alpha   :", args.alpha)
    print("Robot DoF   :", optimizer.robot.dof)
    print("Target DoF  :", len(target_joint_names))
    print("Fixed DoF   :", n_fixed)
    print("Fixed names :", fixed_joint_names)
    print("Fixed qpos  :", fixed_qpos.tolist())
    print("Initial qpos:", initial_qpos.tolist())
    print("=" * 72)

    total = 0
    valid = 0
    invalid = 0

    with input_path.open("r") as fin, \
            output_path.open("w") as fout:

        for line_number, line in enumerate(fin, start=1):

            line = line.strip()

            if not line:
                continue

            src = json.loads(line)

            total += 1

            tracking_valid = bool(
                src.get("tracking_valid", False)
            )

            ref = src.get(
                "reference_vectors_aligned"
            )

            out = {
                "seq": src.get("seq"),
                "timestamp": src.get("timestamp"),
                "capture_monotonic_s":
                    src.get("capture_monotonic_s"),
                "tracking_valid": tracking_valid,

                "source_file": input_path.name,

                "low_pass_alpha":
                    float(args.alpha),

                # Preserve Gate 3A output for replay validation.
                "q_gate3a_logged":
                    src.get("positions_rad"),

                "reference_vectors_aligned":
                    ref,

                "robot_joint_names":
                    robot_joint_names,

                "target_joint_names":
                    target_joint_names,

                "fixed_joint_names":
                    fixed_joint_names,
            }

            # --------------------------------------------------
            # Invalid frame
            #
            # Do NOT update:
            #   - optimizer last_qpos
            #   - LPFilter state
            #
            # This preserves the actual dropout timeline.
            # --------------------------------------------------

            if (not tracking_valid) or (ref is None):

                invalid += 1

                out.update({
                    "q_optimizer_target": None,
                    "q_robot_prefilter": None,
                    "q_robot_output": None,

                    "joint_limit_margin_rad": None,
                    "joint_limit_lower_margin_rad": None,
                    "joint_limit_upper_margin_rad": None,

                    "retarget_ms": None,
                })

                fout.write(
                    json.dumps(out) + "\n"
                )

                continue

            # --------------------------------------------------
            # Valid frame
            # --------------------------------------------------

            ref_value = np.asarray(
                ref,
                dtype=np.float32,
            )

            if ref_value.shape != (10, 3):
                raise ValueError(
                    f"Line {line_number}: "
                    f"expected aligned vectors shape (10,3), "
                    f"got {ref_value.shape}"
                )

            if not np.all(np.isfinite(ref_value)):
                raise ValueError(
                    f"Line {line_number}: "
                    "reference_vectors_aligned "
                    "contains NaN/Inf"
                )

            tic = time.perf_counter()

            # Returned value is AFTER optional LPFilter.
            q_robot_output = retargeter.retarget(
                ref_value,
                fixed_qpos=fixed_qpos,
            )

            retarget_ms = (
                time.perf_counter() - tic
            ) * 1000.0

            # SeqRetargeting stores raw optimizer result here
            # BEFORE constructing the full robot q and LPFilter.
            q_optimizer = np.asarray(
                retargeter.last_qpos,
                dtype=np.float64,
            ).copy()

            # Reconstruct full robot q BEFORE LPFilter.
            q_robot_prefilter = (
                reconstruct_prefilter_qpos(
                    retargeter,
                    fixed_qpos,
                )
            )

            q_robot_output = np.asarray(
                q_robot_output,
                dtype=np.float64,
            )

            if q_optimizer.shape != (
                len(target_joint_names),
            ):
                raise RuntimeError(
                    "Unexpected optimizer q shape: "
                    f"{q_optimizer.shape}"
                )

            if q_robot_output.shape != (
                optimizer.robot.dof,
            ):
                raise RuntimeError(
                    "Unexpected robot q shape: "
                    f"{q_robot_output.shape}"
                )

            # --------------------------------------------------
            # Joint-limit margins for optimized joints
            # --------------------------------------------------

            lower_margin = (
                q_optimizer - target_limits[:, 0]
            )

            upper_margin = (
                target_limits[:, 1] - q_optimizer
            )

            limit_margin = np.minimum(
                lower_margin,
                upper_margin,
            )

            out.update({
                "q_optimizer_target":
                    to_list(q_optimizer),

                "q_robot_prefilter":
                    to_list(q_robot_prefilter),

                "q_robot_output":
                    to_list(q_robot_output),

                "joint_limit_margin_rad":
                    to_list(limit_margin),

                "joint_limit_lower_margin_rad":
                    to_list(lower_margin),

                "joint_limit_upper_margin_rad":
                    to_list(upper_margin),

                "retarget_ms":
                    float(retarget_ms),
            })

            fout.write(
                json.dumps(out) + "\n"
            )

            valid += 1

    print()
    print("[DONE]")
    print("total   =", total)
    print("valid   =", valid)
    print("invalid =", invalid)
    print("saved   =", output_path)


if __name__ == "__main__":
    main()