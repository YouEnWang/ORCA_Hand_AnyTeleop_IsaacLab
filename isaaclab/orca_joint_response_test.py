#!/usr/bin/env python3

"""
Gate 2B - ORCA Isaac joint-response characterization.

Experiments
-----------
1. static
   Constant zero joint targets after a settling period.
   Used to estimate intrinsic simulator/controller noise floor.

2. step
   Sequential +/- step tests for selected joints.

   control-mode=direct:
       q_desired == q_command
       Characterizes the Isaac articulation/controller response.

   control-mode=slew:
       q_desired -> software slew-rate limiter -> q_command
       Characterizes the current teleoperation command stack.

3. sine
   Sequential sinusoidal commands at specified frequencies.
   Used to estimate gain, phase lag, equivalent delay, and tracking RMSE.

Important
---------
All experiment timing is driven by Isaac simulation time, not wall-clock
time. This avoids the real-time-factor issue observed during Gate 2A.

Output
------
A compressed NPZ containing:
    time
    q_desired
    q_command
    q_actual
    joint_names
    phase
    event_id
    active_joint_index
    event_sign
    frequency_hz

and a sidecar JSON file containing experiment metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


# ======================================================================
# CLI
# ======================================================================

parser = argparse.ArgumentParser(
    description="ORCA Gate 2B joint-response characterization."
)

parser.add_argument(
    "--experiment",
    choices=["static", "step", "sine"],
    required=True,
)

parser.add_argument(
    "--output",
    type=Path,
    required=True,
)

parser.add_argument(
    "--dt",
    type=float,
    default=1.0 / 120.0,
)

parser.add_argument(
    "--joint-index",
    default="all",
    help=(
        "'all', one integer such as '6', or comma-separated "
        "indices such as '0,3,6'."
    ),
)

parser.add_argument(
    "--list-joints",
    action="store_true",
)

parser.add_argument(
    "--control-mode",
    choices=["direct", "slew"],
    default="direct",
)

parser.add_argument(
    "--max-joint-speed-rad-s",
    type=float,
    default=2.0,
)

# Static ---------------------------------------------------------------

parser.add_argument(
    "--static-settle-s",
    type=float,
    default=5.0,
)

parser.add_argument(
    "--static-duration-s",
    type=float,
    default=20.0,
)

# Step -----------------------------------------------------------------

parser.add_argument(
    "--amplitude-deg",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--pre-s",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--step-s",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--post-s",
    type=float,
    default=3.0,
)

# Sine -----------------------------------------------------------------

parser.add_argument(
    "--sine-amplitude-deg",
    type=float,
    default=5.0,
)

parser.add_argument(
    "--frequencies",
    default="0.5,1.0,2.0,4.0",
)

parser.add_argument(
    "--cycles",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--min-sine-duration-s",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--sine-pre-s",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--sine-post-s",
    type=float,
    default=2.0,
)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)

simulation_app = app_launcher.app


# ======================================================================
# Isaac imports - must come after AppLauncher
# ======================================================================

import numpy as np
import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation

from orca_hand_cfg import ORCA_HAND_CFG


# ======================================================================
# Helpers
# ======================================================================

def parse_joint_indices(
    spec: str,
    joint_count: int,
) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(joint_count))

    output: list[int] = []

    for token in spec.split(","):
        token = token.strip()

        if not token:
            continue

        idx = int(token)

        if idx < 0 or idx >= joint_count:
            raise ValueError(
                f"Joint index {idx} outside [0, {joint_count - 1}]"
            )

        output.append(idx)

    if not output:
        raise ValueError("No valid joint indices selected.")

    return output


def parse_frequencies(
    text: str,
) -> list[float]:
    output = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if value <= 0.0:
            raise ValueError(
                f"Frequency must be > 0 Hz, got {value}"
            )

        output.append(value)

    if not output:
        raise ValueError("No frequencies supplied.")

    return output


def seconds_to_steps(
    seconds: float,
    dt: float,
) -> int:
    return max(
        1,
        int(
            round(
                seconds / dt
            )
        ),
    )


def to_jsonable(
    value,
):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        return float(value)

    return value


# ======================================================================
# Create simulation
# ======================================================================

sim_cfg = sim_utils.SimulationCfg(
    dt=args.dt,
    device=args.device,
)

sim = sim_utils.SimulationContext(
    sim_cfg
)

sim.set_camera_view(
    eye=[1.1, 0.8, 0.7],
    target=[0.0, 0.0, 0.1],
)

light_cfg = sim_utils.DomeLightCfg(
    intensity=2000.0,
)

light_cfg.func(
    "/World/DomeLight",
    light_cfg,
)

hand_cfg = ORCA_HAND_CFG.replace(
    prim_path="/World/ORCAHand"
)

hand = Articulation(
    cfg=hand_cfg
)

sim.reset()

dt = float(args.dt)

joint_names = list(
    hand.joint_names
)

joint_count = len(
    joint_names
)

print()
print("=" * 80)
print("ORCA Gate 2B Joint Response Test")
print("=" * 80)

print(
    f"Experiment     : {args.experiment}"
)

print(
    f"Simulation dt : {dt:.9f} s"
)

print(
    f"Simulation Hz : {1.0 / dt:.3f} Hz"
)

print(
    f"Joint count    : {joint_count}"
)

print()

for idx, name in enumerate(
    joint_names
):
    print(
        f"[{idx:02d}] {name}"
    )

print()

if joint_count != 17:
    raise RuntimeError(
        f"Expected 17 joints, got {joint_count}"
    )

if args.list_joints:
    simulation_app.close()
    sys.exit(0)


selected_indices = parse_joint_indices(
    args.joint_index,
    joint_count,
)


# ======================================================================
# Soft limits
# ======================================================================

if hasattr(
    hand.data,
    "soft_joint_pos_limits",
):
    limits = (
        hand.data
        .soft_joint_pos_limits[0]
        .clone()
    )

    lower_limits = limits[:, 0]
    upper_limits = limits[:, 1]

else:
    lower_limits = torch.full(
        (joint_count,),
        -float("inf"),
        device=hand.device,
    )

    upper_limits = torch.full(
        (joint_count,),
        float("inf"),
        device=hand.device,
    )


def clamp_target(
    q: torch.Tensor,
) -> torch.Tensor:
    return torch.maximum(
        torch.minimum(
            q,
            upper_limits.unsqueeze(0),
        ),
        lower_limits.unsqueeze(0),
    )


# ======================================================================
# Initial target / command
# ======================================================================

q_zero = torch.zeros_like(
    hand.data.joint_pos
)

q_desired = q_zero.clone()
q_command = q_zero.clone()


# ======================================================================
# Logging storage
# ======================================================================

log_time: list[float] = []

log_desired: list[np.ndarray] = []
log_command: list[np.ndarray] = []
log_actual: list[np.ndarray] = []

log_phase: list[str] = []

log_event_id: list[int] = []
log_active_joint: list[int] = []
log_event_sign: list[int] = []
log_frequency: list[float] = []

event_metadata: list[dict] = []

simulation_time = 0.0


def apply_slew_limit(
    desired: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    max_delta = (
        args.max_joint_speed_rad_s
        * dt
    )

    delta = torch.clamp(
        desired - command,
        min=-max_delta,
        max=max_delta,
    )

    return command + delta


def step_simulation(
    desired: torch.Tensor,
    *,
    phase: str,
    event_id: int,
    active_joint_index: int,
    event_sign: int = 0,
    frequency_hz: float = 0.0,
    record: bool = True,
) -> None:
    """
    Advance Isaac by exactly one simulation step.
    """

    global q_desired
    global q_command
    global simulation_time

    q_desired = clamp_target(
        desired.clone()
    )

    if args.control_mode == "direct":
        q_command = (
            q_desired.clone()
        )

    elif args.control_mode == "slew":
        q_command = apply_slew_limit(
            q_desired,
            q_command,
        )

        q_command = clamp_target(
            q_command
        )

    else:
        raise RuntimeError(
            f"Unknown control mode: {args.control_mode}"
        )

    hand.set_joint_position_target(
        q_command
    )

    hand.write_data_to_sim()

    sim.step()

    hand.update(
        dt
    )

    simulation_time += dt

    if not record:
        return

    log_time.append(
        simulation_time
    )

    log_desired.append(
        q_desired[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    log_command.append(
        q_command[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    log_actual.append(
        hand.data.joint_pos[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    log_phase.append(
        phase
    )

    log_event_id.append(
        event_id
    )

    log_active_joint.append(
        active_joint_index
    )

    log_event_sign.append(
        event_sign
    )

    log_frequency.append(
        frequency_hz
    )


def run_constant(
    target: torch.Tensor,
    duration_s: float,
    *,
    phase: str,
    event_id: int,
    active_joint_index: int,
    event_sign: int = 0,
    frequency_hz: float = 0.0,
    record: bool = True,
) -> None:
    steps = seconds_to_steps(
        duration_s,
        dt,
    )

    for _ in range(steps):
        step_simulation(
            target,
            phase=phase,
            event_id=event_id,
            active_joint_index=active_joint_index,
            event_sign=event_sign,
            frequency_hz=frequency_hz,
            record=record,
        )


# ======================================================================
# Experiment 1: Static
# ======================================================================

if args.experiment == "static":

    print(
        "[STATIC] Settling..."
    )

    run_constant(
        q_zero,
        args.static_settle_s,
        phase="settle",
        event_id=-1,
        active_joint_index=-1,
        record=False,
    )

    print(
        "[STATIC] Recording constant zero target..."
    )

    run_constant(
        q_zero,
        args.static_duration_s,
        phase="static",
        event_id=0,
        active_joint_index=-1,
        record=True,
    )

    event_metadata.append(
        {
            "event_id": 0,
            "type": "static",
            "duration_s":
                args.static_duration_s,
        }
    )


# ======================================================================
# Experiment 2: Step response
# ======================================================================

elif args.experiment == "step":

    amplitude_rad = math.radians(
        args.amplitude_deg
    )

    event_id = 0

    for joint_index in selected_indices:

        for sign in (+1, -1):

            name = joint_names[
                joint_index
            ]

            print()
            print(
                f"[STEP] event={event_id} "
                f"joint={joint_index}:{name} "
                f"sign={sign:+d}"
            )

            # --------------------------------------------------
            # Pre-zero
            # --------------------------------------------------

            run_constant(
                q_zero,
                args.pre_s,
                phase="pre",
                event_id=event_id,
                active_joint_index=joint_index,
                event_sign=sign,
            )

            # --------------------------------------------------
            # Step target
            # --------------------------------------------------

            step_target = (
                q_zero.clone()
            )

            step_target[
                0,
                joint_index,
            ] = (
                sign
                * amplitude_rad
            )

            step_target = clamp_target(
                step_target
            )

            actual_target = float(
                step_target[
                    0,
                    joint_index,
                ].item()
            )

            run_constant(
                step_target,
                args.step_s,
                phase="step",
                event_id=event_id,
                active_joint_index=joint_index,
                event_sign=sign,
            )

            # --------------------------------------------------
            # Return to zero
            # --------------------------------------------------

            run_constant(
                q_zero,
                args.post_s,
                phase="post",
                event_id=event_id,
                active_joint_index=joint_index,
                event_sign=sign,
            )

            event_metadata.append(
                {
                    "event_id":
                        event_id,

                    "type":
                        "step",

                    "joint_index":
                        joint_index,

                    "joint_name":
                        name,

                    "sign":
                        sign,

                    "requested_amplitude_deg":
                        args.amplitude_deg,

                    "actual_target_rad":
                        actual_target,
                }
            )

            event_id += 1


# ======================================================================
# Experiment 3: Sinusoidal response
# ======================================================================

elif args.experiment == "sine":

    amplitude_rad = math.radians(
        args.sine_amplitude_deg
    )

    frequencies = parse_frequencies(
        args.frequencies
    )

    event_id = 0

    for joint_index in selected_indices:

        name = joint_names[
            joint_index
        ]

        for frequency_hz in frequencies:

            duration_s = max(
                args.cycles
                / frequency_hz,
                args.min_sine_duration_s,
            )

            print()
            print(
                f"[SINE] event={event_id} "
                f"joint={joint_index}:{name} "
                f"f={frequency_hz:.3f} Hz "
                f"duration={duration_s:.3f} s"
            )

            # --------------------------------------------------
            # Pre-zero
            # --------------------------------------------------

            run_constant(
                q_zero,
                args.sine_pre_s,
                phase="pre",
                event_id=event_id,
                active_joint_index=joint_index,
                frequency_hz=frequency_hz,
            )

            # --------------------------------------------------
            # Sine excitation
            # --------------------------------------------------

            sine_steps = seconds_to_steps(
                duration_s,
                dt,
            )

            for sample_index in range(
                sine_steps
            ):
                t_rel = (
                    sample_index
                    * dt
                )

                value = (
                    amplitude_rad
                    * math.sin(
                        2.0
                        * math.pi
                        * frequency_hz
                        * t_rel
                    )
                )

                sine_target = (
                    q_zero.clone()
                )

                sine_target[
                    0,
                    joint_index,
                ] = value

                step_simulation(
                    sine_target,
                    phase="sine",
                    event_id=event_id,
                    active_joint_index=joint_index,
                    frequency_hz=frequency_hz,
                )

            # --------------------------------------------------
            # Post-zero
            # --------------------------------------------------

            run_constant(
                q_zero,
                args.sine_post_s,
                phase="post",
                event_id=event_id,
                active_joint_index=joint_index,
                frequency_hz=frequency_hz,
            )

            event_metadata.append(
                {
                    "event_id":
                        event_id,

                    "type":
                        "sine",

                    "joint_index":
                        joint_index,

                    "joint_name":
                        name,

                    "frequency_hz":
                        frequency_hz,

                    "requested_amplitude_deg":
                        args.sine_amplitude_deg,

                    "duration_s":
                        duration_s,
                }
            )

            event_id += 1


else:
    raise RuntimeError(
        f"Unsupported experiment: {args.experiment}"
    )


# ======================================================================
# Save dataset
# ======================================================================

args.output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

time_array = np.asarray(
    log_time,
    dtype=np.float64,
)

desired_array = np.asarray(
    log_desired,
    dtype=np.float64,
)

command_array = np.asarray(
    log_command,
    dtype=np.float64,
)

actual_array = np.asarray(
    log_actual,
    dtype=np.float64,
)

np.savez_compressed(
    args.output,
    time=time_array,
    q_desired=desired_array,
    q_command=command_array,
    q_actual=actual_array,
    joint_names=np.asarray(
        joint_names,
    ),
    phase=np.asarray(
        log_phase,
    ),
    event_id=np.asarray(
        log_event_id,
        dtype=np.int64,
    ),
    active_joint_index=np.asarray(
        log_active_joint,
        dtype=np.int64,
    ),
    event_sign=np.asarray(
        log_event_sign,
        dtype=np.int64,
    ),
    frequency_hz=np.asarray(
        log_frequency,
        dtype=np.float64,
    ),
)

metadata = {
    "experiment":
        args.experiment,

    "control_mode":
        args.control_mode,

    "dt":
        dt,

    "sample_rate_hz":
        1.0 / dt,

    "max_joint_speed_rad_s":
        args.max_joint_speed_rad_s,

    "joint_count":
        joint_count,

    "joint_names":
        joint_names,

    "selected_joint_indices":
        selected_indices,

    "events":
        event_metadata,
}

metadata_path = (
    args.output
    .with_suffix(".json")
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        default=to_jsonable,
    )

print()
print("=" * 80)
print("[PASS] Gate 2B experiment completed.")
print(
    f"[DATA] {args.output}"
)
print(
    f"[META] {metadata_path}"
)
print(
    f"[ROWS] {len(time_array)}"
)
print("=" * 80)

simulation_app.close()