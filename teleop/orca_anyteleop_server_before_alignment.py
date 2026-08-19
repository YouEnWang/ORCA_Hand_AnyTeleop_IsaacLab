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

# ==============================================================
# MediaPipe hand landmark names
#
# Used only for diagnostics.
# The indices follow the standard MediaPipe Hands convention.
# ==============================================================

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
    
    # ============================================================
    # ORCA retargeting initialization
    # ============================================================

    robot = optimizer.robot

    joint_limits = np.asarray(
        robot.joint_limits,
        dtype=np.float32,
    )

    # Start from zero whenever zero is inside the joint range.
    # If zero is outside a joint range, clamp to the nearest valid value.
    initial_qpos = np.zeros(
        robot.dof,
        dtype=np.float32,
    )

    initial_qpos = np.clip(
        initial_qpos,
        joint_limits[:, 0],
        joint_limits[:, 1],
    )

    # Keep ORCA wrist fixed.
    wrist_index = joint_names.index(
        EXPECTED_WRIST_JOINT
    )

    initial_qpos[wrist_index] = np.clip(
        args.wrist_position,
        joint_limits[wrist_index, 0],
        joint_limits[wrist_index, 1],
    )

    # Override SeqRetargeting's default joint-limit-midpoint initialization.
    retargeting.set_qpos(
        initial_qpos
    )

    print()
    print("=" * 80)
    print("ORCA retargeting initial pose")
    print("=" * 80)

    for i, name in enumerate(
        joint_names
    ):
        print(
            f"[{i:02d}] "
            f"{name:<65} "
            f"q0={initial_qpos[i]:+.4f} "
            f"range="
            f"[{joint_limits[i, 0]:+.4f}, "
            f"{joint_limits[i, 1]:+.4f}]"
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
    
    # ==============================================================
    # Human reference-vector configuration diagnostics
    # ==============================================================

    print()
    print("=" * 80)
    print("Human reference-vector mapping")
    print("=" * 80)

    for vector_index, (
        origin_index,
        task_index,
    ) in enumerate(
        zip(
            origin_indices,
            task_indices,
        )
    ):

        origin_index = int(
            origin_index
        )

        task_index = int(
            task_index
        )

        origin_name = (
            MEDIAPIPE_LANDMARK_NAMES.get(
                origin_index,
                f"landmark_{origin_index}",
            )
        )

        task_name = (
            MEDIAPIPE_LANDMARK_NAMES.get(
                task_index,
                f"landmark_{task_index}",
            )
        )

        print(
            f"[HV{vector_index:02d}] "
            f"MP{origin_index:02d} "
            f"{origin_name:<12} "
            f"-> "
            f"MP{task_index:02d} "
            f"{task_name}"
        )

    print()

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

    # ==============================================================
    # Runtime diagnostics
    # ==============================================================

    seq = 0

    # Camera / MediaPipe statistics
    captured_count = 0                  # 成功從 D435i 讀到幾張 frame
    read_failure_count = 0              # camera read 失敗幾次
    valid_detection_count = 0           # MediaPipe 成功抓到指定手
    invalid_detection_count = 0         # MediaPipe 沒抓到手

    # Retargeting / network statistics
    retarget_success_count = 0          # VectorOptimizer 成功產生 qpos
    retarget_failure_count = 0          # optimizer 產生 NaN/Inf
    sent_valid_count = 0                # 發出有效 robot command
    sent_invalid_count = 0              # 發出 tracking_valid=False

    # Latest retargeting diagnostics
    previous_qpos = None
    latest_qpos = None

    latest_q_delta_max = 0.0            # 相鄰兩個 qpos 最大關節變化
    latest_retarget_ms = 0.0            # 最近一幀 optimizer 計算時間

    # ==============================================================
    # Human reference-vector diagnostics
    # ==============================================================

    # Latest human reference vectors actually passed into
    # dex-retargeting VectorOptimizer.
    latest_reference_vectors = None

    # Reference vectors from the previous valid camera frame.
    previous_reference_vectors = None

    # Per-vector length of the latest frame.
    latest_reference_lengths = None

    # Change in vector length from the immediately previous frame.
    latest_reference_delta_lengths = None

    # Full 3-D vector change from the immediately previous frame:
    #
    # || v_t - v_(t-1) ||
    latest_reference_delta_vectors = None
    
    status_timer = time.monotonic()
    
    # Overall runtime timer for experiment alignment.
    run_start_time = time.monotonic()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                
                read_failure_count += 1
                
                print(
                    "[WARNING] "
                    "Failed to read camera frame."
                )

                time.sleep(0.01)

                continue
            
            # 成功取得一張 camera frame
            captured_count += 1
            
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
                
                invalid_detection_count += 1
                
                send_invalid_tracking_packet(
                    sock,
                    args.host,
                    args.port,
                    seq,
                )

                sent_invalid_count += 1
                
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
                
                # ----------------------------------------------------------
                # Even when tracking is invalid, print diagnostics
                # once per second.
                # ----------------------------------------------------------

                now = time.monotonic()

                if now - status_timer >= 1.0:

                    valid_ratio = (
                        100.0
                        * valid_detection_count
                        / max(captured_count, 1)
                    )

                    print(
                        f"[SERVER] "
                        f"camera={captured_count:3d}/s  "
                        f"read_fail={read_failure_count:3d}  "
                        f"valid={valid_detection_count:3d}  "
                        f"invalid={invalid_detection_count:3d}  "
                        f"ratio={valid_ratio:6.2f}%  "
                        f"retarget_ok={retarget_success_count:3d}  "
                        f"retarget_fail={retarget_failure_count:3d}  "
                        f"sent_valid={sent_valid_count:3d}  "
                        f"sent_invalid={sent_invalid_count:3d}"
                    )

                    captured_count = 0
                    read_failure_count = 0
                    valid_detection_count = 0
                    invalid_detection_count = 0

                    retarget_success_count = 0
                    retarget_failure_count = 0
                    sent_valid_count = 0
                    sent_invalid_count = 0

                    status_timer = now
                
                continue
            
            # 到這裡代表 hand detection 成功
            valid_detection_count += 1
            
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
            
            # ==============================================================
            # Human reference-vector diagnostics
            #
            # IMPORTANT:
            # These diagnostics do NOT modify reference_vectors.
            # The original vectors are still passed directly to
            # dex-retargeting below.
            # ==============================================================

            latest_reference_vectors = (
                reference_vectors.copy()
            )

            latest_reference_lengths = (
                np.linalg.norm(
                    latest_reference_vectors,
                    axis=1,
                )
            )


            if previous_reference_vectors is None:

                latest_reference_delta_lengths = (
                    np.zeros_like(
                        latest_reference_lengths
                    )
                )

                latest_reference_delta_vectors = (
                    np.zeros_like(
                        latest_reference_lengths
                    )
                )

            else:

                previous_lengths = (
                    np.linalg.norm(
                        previous_reference_vectors,
                        axis=1,
                    )
                )

                # Signed change of vector magnitude.
                #
                # Negative:
                #   current vector is shorter than previous frame.
                #
                # Positive:
                #   current vector is longer than previous frame.
                latest_reference_delta_lengths = (
                    latest_reference_lengths
                    -
                    previous_lengths
                )

                # Full 3-D vector displacement.
                #
                # This catches orientation changes even when
                # vector length stays almost unchanged.
                latest_reference_delta_vectors = (
                    np.linalg.norm(
                        latest_reference_vectors
                        -
                        previous_reference_vectors,
                        axis=1,
                    )
                )


            previous_reference_vectors = (
                latest_reference_vectors.copy()
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
            
            latest_retarget_ms = float(
                retarget_ms
            )
            
            # --------------------------------------------------------------
            # Validate optimizer output
            # --------------------------------------------------------------

            if not np.all(
                np.isfinite(qpos)
            ):

                retarget_failure_count += 1
                
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

                sent_invalid_count += 1
                
                seq += 1

                continue
            
            # Retargeting output is valid
            retarget_success_count += 1

            latest_qpos = qpos.copy()
            
            # --------------------------------------------------------------
            # Measure frame-to-frame joint-target change
            # --------------------------------------------------------------

            if previous_qpos is None:

                latest_q_delta_max = 0.0

            else:

                latest_q_delta_max = float(
                    np.max(
                        np.abs(
                            qpos
                            -
                            previous_qpos
                        )
                    )
                )


            previous_qpos = qpos.copy()
            
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
            
            sent_valid_count += 1

            seq += 1
            
            # ==============================================================
            # Print server status once per second
            # ==============================================================

            now = time.monotonic()

            if now - status_timer >= 1.0:

                valid_ratio = (
                    100.0
                    * valid_detection_count
                    / max(captured_count, 1)
                )

                if latest_qpos is not None:

                    q_min = float(
                        np.min(latest_qpos)
                    )

                    q_max = float(
                        np.max(latest_qpos)
                    )

                    q_text = (
                        f"[{q_min:+.3f}, "
                        f"{q_max:+.3f}]"
                    )

                else:

                    q_text = "N/A"

                print(
                    f"[SERVER] "
                    f"camera={captured_count:3d}/s  "
                    f"read_fail={read_failure_count:3d}  "
                    f"valid={valid_detection_count:3d}  "
                    f"invalid={invalid_detection_count:3d}  "
                    f"ratio={valid_ratio:6.2f}%  "
                    f"retarget_ok={retarget_success_count:3d}  "
                    f"retarget_fail={retarget_failure_count:3d}  "
                    f"sent={sent_valid_count:3d}  "
                    f"retarget={latest_retarget_ms:6.2f}ms  "
                    f"q={q_text}  "
                    f"dq_max={latest_q_delta_max:.4f}  "
                    f"seq={seq}"
                )
                
                # ==============================================================
                # Human reference-vector diagnostics
                # ==============================================================

                if (
                    latest_reference_vectors is not None
                    and latest_reference_lengths is not None
                ):

                    elapsed_time = (
                        time.monotonic()
                        -
                        run_start_time
                    )

                    print(
                        f"[HVEC] t={elapsed_time:6.2f}s"
                    )

                    for vector_index in range(
                        len(latest_reference_vectors)
                    ):

                        origin_index = int(
                            origin_indices[
                                vector_index
                            ]
                        )

                        task_index = int(
                            task_indices[
                                vector_index
                            ]
                        )

                        origin_name = (
                            MEDIAPIPE_LANDMARK_NAMES.get(
                                origin_index,
                                f"landmark_{origin_index}",
                            )
                        )

                        task_name = (
                            MEDIAPIPE_LANDMARK_NAMES.get(
                                task_index,
                                f"landmark_{task_index}",
                            )
                        )

                        vx = float(
                            latest_reference_vectors[
                                vector_index,
                                0
                            ]
                        )

                        vy = float(
                            latest_reference_vectors[
                                vector_index,
                                1
                            ]
                        )

                        vz = float(
                            latest_reference_vectors[
                                vector_index,
                                2
                            ]
                        )

                        length = float(
                            latest_reference_lengths[
                                vector_index
                            ]
                        )

                        delta_length = float(
                            latest_reference_delta_lengths[
                                vector_index
                            ]
                        )

                        delta_vector = float(
                            latest_reference_delta_vectors[
                                vector_index
                            ]
                        )

                        print(
                            f"  [HV{vector_index:02d}] "
                            f"{origin_name:>10} "
                            f"-> "
                            f"{task_name:<12} "
                            f"v="
                            f"[{vx:+.5f}, "
                            f"{vy:+.5f}, "
                            f"{vz:+.5f}]  "
                            f"L={length:.5f}  "
                            f"dL={delta_length:+.6f}  "
                            f"dV={delta_vector:.6f}"
                        )
                
                # ==============================================================
                # Per-joint qpos diagnostics
                # 方便分析到底是哪根手指，哪個 joint，撞哪個 limit
                # ==============================================================

                if latest_qpos is not None:

                    elapsed_time = (
                        time.monotonic()
                        - run_start_time
                    )

                    print(
                        f"[QPOS] t={elapsed_time:6.2f}s"
                    )

                    # dex-retargeting expands optimizer bounds by a very
                    # small epsilon, so use a small tolerance when checking
                    # whether a joint is effectively at a limit.
                    limit_tolerance = 0.003

                    for joint_index, (
                        joint_name,
                        joint_value,
                    ) in enumerate(
                        zip(
                            joint_names,
                            latest_qpos,
                        )
                    ):

                        value = float(
                            joint_value
                        )

                        lower = float(
                            joint_limits[
                                joint_index,
                                0
                            ]
                        )

                        upper = float(
                            joint_limits[
                                joint_index,
                                1
                            ]
                        )

                        joint_range = (
                            upper - lower
                        )

                        # Normalized location inside joint range.
                        #
                        # 0.0 -> lower limit
                        # 0.5 -> center
                        # 1.0 -> upper limit
                        if joint_range > 1e-8:

                            normalized = (
                                (value - lower)
                                / joint_range
                            )

                        else:

                            normalized = 0.0

                        # Detect saturation.
                        if (
                            value
                            <= lower + limit_tolerance
                        ):

                            limit_state = (
                                "<<< LOWER_LIMIT"
                            )

                        elif (
                            value
                            >= upper - limit_tolerance
                        ):

                            limit_state = (
                                "<<< UPPER_LIMIT"
                            )

                        else:

                            limit_state = ""

                        print(
                            f"  [{joint_index:02d}] "
                            f"{joint_name:<65} "
                            f"q={value:+.4f}  "
                            f"norm={normalized:6.3f}  "
                            f"range="
                            f"[{lower:+.4f}, "
                            f"{upper:+.4f}]  "
                            f"{limit_state}"
                        )

                # Reset only the one-second counters.
                # Do NOT reset previous_qpos/latest_qpos.
                captured_count = 0
                read_failure_count = 0
                valid_detection_count = 0
                invalid_detection_count = 0

                retarget_success_count = 0
                retarget_failure_count = 0
                sent_valid_count = 0
                sent_invalid_count = 0

                status_timer = now

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
