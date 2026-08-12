#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AnyTeleop-style ORCA Hand teleoperation server.

Pipeline:

Camera
  ↓
MediaPipe SingleHandDetector
  ↓
21 x 3 wrist-centered hand landmarks
  ↓
Human task-space vectors
  ↓
dex-retargeting VectorOptimizer
  ↓
17-DoF ORCA qpos
  ↓
UDP
  ↓
Isaac Lab ORCA client

This server contains no Isaac Sim dependencies.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dex_retargeting.retargeting_config import (
    RetargetingConfig,
)


# Reuse the hand detector shipped with the
# pinned dex-retargeting / AnyTeleop example.
DEX_EXAMPLE_DIR = (
    "/opt/dex-retargeting/"
    "example/vector_retargeting"
)

if DEX_EXAMPLE_DIR not in sys.path:
    sys.path.insert(
        0,
        DEX_EXAMPLE_DIR,
    )

from single_hand_detector import SingleHandDetector


EXPECTED_WRIST_JOINT = (
    "R-Carpals_8d1f1041_to_"
    "TopTower-Model_4a80d30e"
)


def parse_camera_source(
    value: str,
):
    if value.isdigit():
        return int(value)

    return value


def send_invalid_tracking_packet(
    sock: socket.socket,
    host: str,
    port: int,
    seq: int,
):

    packet = {
        "seq": seq,
        "timestamp": time.time(),
        "tracking_valid": False,
        "source": "mediapipe",
    }

    sock.sendto(
        json.dumps(
            packet
        ).encode("utf-8"),
        (
            host,
            port,
        ),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5006,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--hand-type",
        choices=["Right", "Left"],
        default="Right",
    )

    parser.add_argument(
        "--selfie",
        action="store_true",
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    parser.add_argument(
        "--wrist-position",
        type=float,
        default=0.0,
        help="Fixed ORCA wrist position in radians.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Build retargeting
    # --------------------------------------------------------------

    cfg = RetargetingConfig.load_from_file(
        args.config.resolve()
    )

    retargeting = cfg.build()

    optimizer = retargeting.optimizer

    joint_names = list(
        retargeting.joint_names
    )

    fixed_joint_names = (
        optimizer.fixed_joint_names
    )

    if len(joint_names) != 17:
        raise RuntimeError(
            f"Expected 17 ORCA DoF, "
            f"got {len(joint_names)}"
        )

    if fixed_joint_names != [
        EXPECTED_WRIST_JOINT
    ]:
        raise RuntimeError(
            "Unexpected fixed joints:\n"
            f"{fixed_joint_names}"
        )

    fixed_qpos = np.asarray(
        [args.wrist_position],
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # Human detector
    # --------------------------------------------------------------

    detector = SingleHandDetector(
        hand_type=args.hand_type,
        selfie=args.selfie,
    )

    # --------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------

    camera_source = parse_camera_source(
        args.camera
    )

    cap = cv2.VideoCapture(
        camera_source
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera: "
            f"{camera_source}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps,
    )

    # --------------------------------------------------------------
    # UDP
    # --------------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    # --------------------------------------------------------------
    # Retargeting human landmark indices
    # --------------------------------------------------------------

    indices = (
        optimizer.target_link_human_indices
    )

    origin_indices = (
        indices[0, :]
    )

    task_indices = (
        indices[1, :]
    )

    print("=" * 80)
    print("AnyTeleop ORCA Hand Server")
    print("=" * 80)

    print("Camera            :", camera_source)
    print("Hand type         :", args.hand_type)
    print("Selfie            :", args.selfie)
    print("Retargeting       :", optimizer.retargeting_type)
    print("Robot DOF         :", len(joint_names))
    print("Optimized DOF     :", optimizer.opt_dof)
    print("Fixed joints      :", fixed_joint_names)
    print("UDP destination   :", f"{args.host}:{args.port}")

    print()
    print("ORCA robot joint order:")

    for i, name in enumerate(
        joint_names
    ):
        print(
            f"  [{i:02d}] {name}"
        )

    seq = 0

    frame_count = 0

    fps_timer = time.monotonic()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "[WARNING] "
                    "Failed to read camera frame."
                )

                time.sleep(0.01)

                continue

            # OpenCV BGR -> MediaPipe RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                num_box,
                joint_pos,
                keypoint_2d,
                mediapipe_wrist_rot,
            ) = detector.detect(
                rgb
            )

            if num_box == 0:

                send_invalid_tracking_packet(
                    sock,
                    args.host,
                    args.port,
                    seq,
                )

                seq += 1

                if args.show:

                    cv2.putText(
                        frame,
                        "NO HAND",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                    cv2.imshow(
                        "AnyTeleop ORCA",
                        frame,
                    )

                    if (
                        cv2.waitKey(1)
                        & 0xFF
                    ) == ord("q"):
                        break

                continue

            # ------------------------------------------------------
            # This follows the official dex-retargeting
            # vector-retargeting example:
            #
            # ref_vector =
            #     task landmark
            #   - origin landmark
            # ------------------------------------------------------

            reference_vectors = (
                joint_pos[
                    task_indices,
                    :
                ]
                -
                joint_pos[
                    origin_indices,
                    :
                ]
            )

            start_retarget = (
                time.perf_counter()
            )

            qpos = retargeting.retarget(
                reference_vectors,
                fixed_qpos=fixed_qpos,
            )

            retarget_ms = (
                (
                    time.perf_counter()
                    - start_retarget
                )
                * 1000.0
            )

            if not np.all(
                np.isfinite(qpos)
            ):

                print(
                    "[WARNING] "
                    "Retargeting returned "
                    "NaN/Inf."
                )

                send_invalid_tracking_packet(
                    sock,
                    args.host,
                    args.port,
                    seq,
                )

                seq += 1

                continue

            # ------------------------------------------------------
            # Network packet
            # ------------------------------------------------------

            packet = {
                "seq": seq,
                "timestamp": time.time(),
                "tracking_valid": True,
                "joint_names": joint_names,
                "positions_rad": (
                    qpos.astype(float)
                    .tolist()
                ),
                "retarget_ms": (
                    float(retarget_ms)
                ),
                "source": "mediapipe_dex_retargeting",
            }

            sock.sendto(
                json.dumps(
                    packet
                ).encode("utf-8"),
                (
                    args.host,
                    args.port,
                ),
            )

            seq += 1
            frame_count += 1

            # ------------------------------------------------------
            # Console FPS
            # ------------------------------------------------------

            now = time.monotonic()

            elapsed = (
                now - fps_timer
            )

            if elapsed >= 1.0:

                measured_fps = (
                    frame_count
                    / elapsed
                )

                print(
                    f"[RUN] FPS={measured_fps:5.1f}  "
                    f"retarget={retarget_ms:6.2f} ms  "
                    f"seq={seq}"
                )

                frame_count = 0
                fps_timer = now

            # ------------------------------------------------------
            # Optional display
            # ------------------------------------------------------

            if args.show:

                if keypoint_2d is not None:

                    frame = (
                        detector
                        .draw_skeleton_on_image(
                            frame,
                            keypoint_2d,
                        )
                    )

                cv2.putText(
                    frame,
                    f"retarget: "
                    f"{retarget_ms:.2f} ms",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "AnyTeleop ORCA",
                    frame,
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                ) == ord("q"):
                    break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted."
        )

    finally:

        cap.release()

        sock.close()

        if args.show:
            cv2.destroyAllWindows()

        print()
        retargeting.verbose()


if __name__ == "__main__":
    main()
