#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AnyTeleop-style ORCA Hand client for Isaac Lab.

Architecture:

    dex-retargeting container
            |
            | UDP JSON
            v
    this Isaac Lab client
            |
            v
    ORCA Hand v2 Right

The client does NOT:
- run MediaPipe
- run dex-retargeting
- know human landmarks

It only:
1. receives robot qpos
2. maps joints by name
3. validates the command
4. applies joint limits
5. applies a velocity safety limit
6. sends position targets to Isaac Lab
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time


# ----------------------------------------------------------------------
# Isaac Lab application launcher
# ----------------------------------------------------------------------

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

parser.add_argument(
    "--udp-host",
    type=str,
    default="0.0.0.0",
)

parser.add_argument(
    "--udp-port",
    type=int,
    default=5006,
)

parser.add_argument(
    "--timeout",
    type=float,
    default=0.5,
    help="Hold last pose if no valid packet is received for this duration.",
)

parser.add_argument(
    "--max-packet-age",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--max-joint-speed",
    type=float,
    default=2.0,
    help="Command slew-rate limit in rad/s.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


# ----------------------------------------------------------------------
# Isaac imports after SimulationApp launch
# ----------------------------------------------------------------------

import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from orca_hand_cfg import ORCA_HAND_CFG


def sanitize_urdf_name(name: str) -> str:
    """
    Isaac's USD importer replaces unsupported prim/joint-name
    characters such as '-' with '_'.
    """

    return re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        name,
    )


def build_packet_to_isaac_map(
    packet_joint_names: list[str],
    isaac_joint_names: list[str],
) -> list[int]:

    sanitized_packet_names = [
        sanitize_urdf_name(name)
        for name in packet_joint_names
    ]

    if len(set(sanitized_packet_names)) != len(
        sanitized_packet_names
    ):
        raise RuntimeError(
            "Sanitized packet joint names are not unique."
        )

    mapping = []

    for isaac_name in isaac_joint_names:

        if isaac_name not in sanitized_packet_names:

            raise RuntimeError(
                "Cannot map Isaac joint:\n"
                f"  Isaac name: {isaac_name}\n"
                f"  Packet names after sanitization:\n"
                + "\n".join(
                    f"    {name}"
                    for name in sanitized_packet_names
                )
            )

        mapping.append(
            sanitized_packet_names.index(
                isaac_name
            )
        )

    return mapping


def setup_simulation():

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
    )

    sim = SimulationContext(
        sim_cfg
    )

    # Lighting
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )

    light_cfg.func(
        "/World/Light",
        light_cfg,
    )

    # ORCA
    robot = Articulation(
        ORCA_HAND_CFG
    )

    # Camera
    sim.set_camera_view(
        eye=(0.45, 0.35, 0.45),
        target=(0.0, 0.0, 0.4),
    )

    sim.reset()

    robot.update(
        sim_cfg.dt
    )

    return sim, sim_cfg.dt, robot


def main():

    sim, sim_dt, robot = setup_simulation()

    print()
    print("=" * 80)
    print("ORCA AnyTeleop Isaac Client")
    print("=" * 80)

    print("Device        :", robot.device)
    print("Joint count   :", robot.num_joints)
    print("Fixed base    :", robot.is_fixed_base)

    print()
    print("Isaac runtime joint order:")

    for i, name in enumerate(
        robot.joint_names
    ):
        print(
            f"  [{i:02d}] {name}"
        )

    if robot.num_joints != 17:
        raise RuntimeError(
            f"Expected 17 ORCA joints, "
            f"got {robot.num_joints}"
        )

    # --------------------------------------------------------------
    # UDP
    # --------------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    sock.bind(
        (
            args_cli.udp_host,
            args_cli.udp_port,
        )
    )

    sock.setblocking(False)

    print()
    print(
        f"Listening: udp://"
        f"{args_cli.udp_host}:"
        f"{args_cli.udp_port}"
    )

    # --------------------------------------------------------------
    # Initial target
    # --------------------------------------------------------------

    target = (
        robot.data.joint_pos
        .clone()
    )

    desired_target = (
        target.clone()
    )

    # Soft safety limits.
    joint_limits = (
        robot.data.soft_joint_pos_limits[0]
    )

    print()
    print("Soft joint limits:")

    for i, name in enumerate(
        robot.joint_names
    ):
        lower = float(
            joint_limits[i, 0]
        )

        upper = float(
            joint_limits[i, 1]
        )

        print(
            f"  {name:<65} "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}]"
        )

    packet_mapping = None

    last_valid_receive_time = None
    last_seq = -1

    # ==============================================================
    # UDP / Isaac diagnostics
    # ==============================================================

    received_count = 0
    valid_received_count = 0
    invalid_received_count = 0
    json_error_count = 0

    status_timer = time.monotonic()
    
    max_step = (
        args_cli.max_joint_speed
        * sim_dt
    )

    print()
    print(
        "Waiting for AnyTeleop "
        "retargeting packets..."
    )

    try:

        while simulation_app.is_running():

            latest_packet = None

            # ------------------------------------------------------
            # Drain UDP queue.
            # Only use the newest packet to minimize latency.
            # Client 會把 UDP queue 裡舊的 packet 全部讀掉，但只真正執行最新的一個。
            # ------------------------------------------------------

            while True:

                try:

                    data, address = sock.recvfrom(
                        65535
                    )

                    packet = json.loads(
                        data.decode("utf-8")
                    )

                    # Count every packet received from the UDP socket.
                    received_count += 1

                    if bool(
                        packet.get(
                            "tracking_valid",
                            False,
                        )
                    ):

                        valid_received_count += 1

                    else:

                        invalid_received_count += 1

                    # We intentionally keep only the newest packet
                    # for low-latency teleoperation.
                    latest_packet = packet


                except BlockingIOError:

                    break


                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:

                    json_error_count += 1

                    print(
                        "[WARNING] Invalid UDP packet:",
                        exc,
                    )

            # ------------------------------------------------------
            # New command
            # ------------------------------------------------------

            if latest_packet is not None:

                tracking_valid = bool(
                    latest_packet.get(
                        "tracking_valid",
                        False,
                    )
                )

                seq = int(
                    latest_packet.get(
                        "seq",
                        -1,
                    )
                )

                timestamp = float(
                    latest_packet.get(
                        "timestamp",
                        0.0,
                    )
                )

                packet_age = (
                    time.time()
                    - timestamp
                )

                if (
                    tracking_valid
                    and seq > last_seq
                    and packet_age
                    <= args_cli.max_packet_age
                ):

                    names = latest_packet[
                        "joint_names"
                    ]

                    positions = latest_packet[
                        "positions_rad"
                    ]

                    if (
                        len(names) != 17
                        or len(positions) != 17
                    ):
                        print(
                            "[WARNING] Expected "
                            "17 names and positions."
                        )

                    else:

                        # Build name map once.
                        if packet_mapping is None:

                            packet_mapping = (
                                build_packet_to_isaac_map(
                                    names,
                                    robot.joint_names,
                                )
                            )

                            print()
                            print(
                                "[PASS] Packet-to-Isaac "
                                "joint mapping:"
                            )

                            for (
                                isaac_index,
                                packet_index,
                            ) in enumerate(
                                packet_mapping
                            ):

                                print(
                                    f"  Isaac[{isaac_index:02d}] "
                                    f"{robot.joint_names[isaac_index]}"
                                )

                                print(
                                    f"      <- packet[{packet_index:02d}] "
                                    f"{names[packet_index]}"
                                )

                        ordered_positions = [
                            positions[
                                packet_index
                            ]
                            for packet_index
                            in packet_mapping
                        ]

                        desired = torch.tensor(
                            ordered_positions,
                            dtype=torch.float32,
                            device=robot.device,
                        )

                        if torch.isfinite(
                            desired
                        ).all():

                            # Clamp to soft limits.
                            desired = torch.clamp(
                                desired,
                                min=joint_limits[:, 0],
                                max=joint_limits[:, 1],
                            )

                            desired_target[0, :] = (
                                desired
                            )

                            last_valid_receive_time = (
                                time.monotonic()
                            )

                            last_seq = seq

                        else:

                            print(
                                "[WARNING] NaN/Inf "
                                "target rejected."
                            )

            # ------------------------------------------------------
            # Timeout:
            # hold last valid pose.
            # ------------------------------------------------------

            if last_valid_receive_time is not None:

                elapsed = (
                    time.monotonic()
                    - last_valid_receive_time
                )

                if elapsed > args_cli.timeout:
                    # Holding desired_target intentionally.
                    pass

            # ------------------------------------------------------
            # Command slew-rate limiter
            # ------------------------------------------------------

            delta = (
                desired_target
                - target
            )

            delta = torch.clamp(
                delta,
                min=-max_step,
                max=max_step,
            )

            target += delta

            # ------------------------------------------------------
            # Isaac position target
            # ------------------------------------------------------

            robot.set_joint_position_target(
                target
            )

            robot.write_data_to_sim()

            sim.step()

            robot.update(
                sim_dt
            )
            
            # ==============================================================
            # Print client / Isaac status once per second
            # ==============================================================

            now = time.monotonic()

            if now - status_timer >= 1.0:

                actual = (
                    robot.data.joint_pos[0]
                )                           # PhysX 中真正的 ORCA joint position

                command_target = (
                    target[0]
                )                           # 經過現在 client 的 max_joint_speed slew-rate limiter 後，真正送給 Isaac actuator 的 target

                desired = (
                    desired_target[0]
                )                           # AnyTeleop / dex-retargeting 真正想要的姿勢

                tracking_error = (
                    command_target
                    -
                    actual
                )

                command_lag = (
                    desired
                    -
                    command_target
                )

                print(
                    f"[CLIENT] "
                    f"recv={received_count:3d}/s  "
                    f"valid={valid_received_count:3d}  "
                    f"invalid={invalid_received_count:3d}  "
                    f"json_err={json_error_count:2d}  "
                    f"last_seq={last_seq:6d}  "
                    f"desired="
                    f"[{float(desired.min()):+.3f}, "
                    f"{float(desired.max()):+.3f}]  "
                    f"applied="
                    f"[{float(command_target.min()):+.3f}, "
                    f"{float(command_target.max()):+.3f}]  "
                    f"actual="
                    f"[{float(actual.min()):+.3f}, "
                    f"{float(actual.max()):+.3f}]  "
                    f"ctrl_err_max="
                    f"{float(torch.abs(tracking_error).max()):.4f}  "
                    f"slew_remaining="
                    f"{float(torch.abs(command_lag).max()):.4f}"
                )

                received_count = 0
                valid_received_count = 0
                invalid_received_count = 0
                json_error_count = 0

                status_timer = now

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted."
        )

    finally:

        sock.close()


if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()
