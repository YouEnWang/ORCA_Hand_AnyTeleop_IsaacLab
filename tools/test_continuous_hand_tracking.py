#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continuous MediaPipe tracking diagnostic.

Purpose
-------
Verify that the camera + SingleHandDetector can continuously
produce valid hand detections before involving:
- dex-retargeting optimizer
- UDP
- Isaac Lab

Run inside the AnyTeleop Docker container.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2


DEX_EXAMPLE_DIR = (
    "/opt/dex-retargeting/"
    "example/vector_retargeting"
)

if DEX_EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, DEX_EXAMPLE_DIR)

from single_hand_detector import SingleHandDetector


def parse_camera(value: str):
    if value.isdigit():
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        default="/dev/video4",
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
        "--duration",
        type=float,
        default=15.0,
    )

    args = parser.parse_args()

    detector = SingleHandDetector(
        hand_type=args.hand_type,
        selfie=args.selfie,
    )

    camera_source = parse_camera(
        args.camera
    )

    cap = cv2.VideoCapture(
        camera_source
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera: {camera_source}"
        )

    print("=" * 70)
    print("Continuous MediaPipe Tracking Test")
    print("=" * 70)
    print("Camera    :", camera_source)
    print("Hand type :", args.hand_type)
    print("Selfie    :", args.selfie)
    print("Duration  :", args.duration, "s")

    total_frames = 0
    valid_frames = 0
    invalid_frames = 0
    read_failures = 0

    interval_total = 0
    interval_valid = 0
    interval_invalid = 0

    start_time = time.monotonic()
    status_time = start_time

    try:
        while (
            time.monotonic() - start_time
            < args.duration
        ):
            ok, frame = cap.read()

            if not ok:
                read_failures += 1
                time.sleep(0.005)
                continue

            total_frames += 1
            interval_total += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                num_box,
                joint_pos,
                keypoint_2d,
                wrist_rot,
            ) = detector.detect(rgb)

            if (
                num_box > 0
                and joint_pos is not None
            ):
                valid_frames += 1
                interval_valid += 1

                valid = True

            else:
                invalid_frames += 1
                interval_invalid += 1

                valid = False

            now = time.monotonic()

            if now - status_time >= 1.0:
                ratio = (
                    100.0
                    * interval_valid
                    / max(interval_total, 1)
                )

                shape = frame.shape

                print(
                    f"[TRACK] "
                    f"frames={interval_total:3d}/s  "
                    f"valid={interval_valid:3d}  "
                    f"invalid={interval_invalid:3d}  "
                    f"valid_ratio={ratio:6.2f}%  "
                    f"resolution="
                    f"{shape[1]}x{shape[0]}  "
                    f"last_valid={valid}"
                )

                interval_total = 0
                interval_valid = 0
                interval_invalid = 0

                status_time = now

    finally:
        cap.release()

    total_ratio = (
        100.0
        * valid_frames
        / max(total_frames, 1)
    )

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print("Total frames :", total_frames)
    print("Valid        :", valid_frames)
    print("Invalid      :", invalid_frames)
    print("Read failure :", read_failures)
    print(
        "Valid ratio  :",
        f"{total_ratio:.2f}%",
    )

    if total_ratio >= 90.0:
        print(
            "[PASS] Continuous tracking "
            "looks stable."
        )

    elif total_ratio >= 70.0:
        print(
            "[WARNING] Tracking works, "
            "but dropout is significant."
        )

    else:
        print(
            "[FAIL] Tracking is not "
            "stable enough for teleoperation."
        )


if __name__ == "__main__":
    main()