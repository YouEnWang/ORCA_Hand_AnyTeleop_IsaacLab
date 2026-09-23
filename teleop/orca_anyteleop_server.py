#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AnyTeleop-style ORCA Hand teleoperation server.

Purpose
-------
This is the thin orchestration entry point for the online ORCA pipeline.
The previous ~1,800-line implementation mixed initialization, alignment,
recording, UDP, diagnostics, and the real-time loop.  Those support
responsibilities are now isolated, while the online algorithm remains the
same.

Per-frame pipeline
------------------

    OpenCV camera frame
      -> SingleHandDetector / MediaPipe
      -> 21 x 3 wrist-centered landmarks
      -> raw human task-space vectors
      -> Human -> ORCA rotation + scale
      -> dex-retargeting VectorOptimizer
      -> 17-DoF ORCA qpos
      -> Gate 3A JSONL recording
      -> optional UDP -> Isaac Lab

Gate 3A ``--dry-run``
---------------------
``--dry-run`` disables every UDP transmission, but camera capture,
MediaPipe, vector construction, alignment, retargeting, diagnostics, and
JSONL recording all continue unchanged.  Therefore Gate 3A measures the
same upstream path that will later feed Isaac.

Companion modules
-----------------
``orca_server_setup.py``
    CLI, detector, retargeter, alignment selection, camera initialization.
``orca_vector_alignment.py``
    Alignment YAML validation and Human -> ORCA vector transformation.
``orca_gate3a_recorder.py``
    JSONL recording and MediaPipe protobuf -> numeric conversion.
``orca_runtime_diagnostics.py``
    Startup and once-per-second diagnostic presentation.
``orca_udp.py``
    UDP packet schema and transmission.
"""

from __future__ import annotations

import socket
import time

import cv2
import numpy as np

from orca_gate3a_recorder import Gate3ARecorder
from orca_runtime_diagnostics import (
    RuntimeDiagnostics,
    print_alignment_configuration,
    print_camera_configuration,
    print_human_vector_mapping,
    print_initial_pose,
    print_robot_joint_order,
    print_server_configuration,
)
from orca_server_setup import (
    build_argument_parser,
    initialize_alignment,
    initialize_camera,
    initialize_detector,
    initialize_retargeting,
)
from orca_udp import (
    build_valid_tracking_packet,
    send_invalid_tracking_packet,
    send_packet,
)
from orca_vector_alignment import apply_vector_alignment


def main():
    """Run the online AnyTeleop-style ORCA hand pipeline."""

    args = build_argument_parser().parse_args()

    (
        alignment_enabled,
        alignment_rotation,
        alignment_scale,
        alignment_det,
        alignment_orth_error,
    ) = initialize_alignment(args)

    (
        retargeting,
        optimizer,
        joint_names,
        joint_limits,
        initial_qpos,
        fixed_joint_names,
        fixed_qpos,
    ) = initialize_retargeting(args)

    print_initial_pose(
        joint_names=joint_names,
        initial_qpos=initial_qpos,
        joint_limits=joint_limits,
    )

    detector = initialize_detector(args)
    camera_source, cap = initialize_camera(args)
    print_camera_configuration(cap)

    # Keep socket lifetime identical even in dry-run; transmission itself
    # is guarded at every send site.
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    recorder = Gate3ARecorder(args.record_path)

    indices = optimizer.target_link_human_indices
    origin_indices = indices[0, :]
    task_indices = indices[1, :]

    print_human_vector_mapping(
        origin_indices=origin_indices,
        task_indices=task_indices,
    )
    print_server_configuration(
        camera_source=camera_source,
        hand_type=args.hand_type,
        selfie=args.selfie,
        retarget_config=args.config,
        alignment_config=args.alignment_config,
        dry_run=args.dry_run,
        retargeting_type=optimizer.retargeting_type,
        joint_names=joint_names,
        optimized_dof=optimizer.opt_dof,
        fixed_joint_names=fixed_joint_names,
        host=args.host,
        port=args.port,
    )
    print_alignment_configuration(
        alignment_enabled=alignment_enabled,
        alignment_config=args.alignment_config,
        alignment_scale=alignment_scale,
        alignment_det=alignment_det,
        alignment_orth_error=alignment_orth_error,
        alignment_rotation=alignment_rotation,
    )
    print_robot_joint_order(joint_names)

    seq = 0
    run_start_time = time.monotonic()

    diagnostics = RuntimeDiagnostics(
        origin_indices=origin_indices,
        task_indices=task_indices,
        joint_names=joint_names,
        joint_limits=joint_limits,
        alignment_scale=alignment_scale,
        run_start_time=run_start_time,
        status_timer=time.monotonic(),
    )

    try:
        while True:
            # ----------------------------------------------------------
            # Optional fixed experiment duration
            # ----------------------------------------------------------
            if (
                args.duration is not None
                and (
                    time.monotonic() - run_start_time
                ) >= args.duration
            ):
                print(
                    "[INFO] Requested experiment duration reached."
                )
                break

            # ----------------------------------------------------------
            # 1. Camera capture
            # ----------------------------------------------------------
            ok, frame = cap.read()

            if not ok:
                diagnostics.on_read_failure()
                print(
                    "[WARNING] Failed to read camera frame."
                )
                time.sleep(0.01)
                continue

            diagnostics.on_capture_success()

            capture_monotonic_s = (
                time.monotonic() - run_start_time
            )
            capture_wall_s = time.time()

            # ----------------------------------------------------------
            # 2. MediaPipe / SingleHandDetector
            # ----------------------------------------------------------
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                num_box,
                joint_pos,
                keypoint_2d,
                mediapipe_wrist_rot,
            ) = detector.detect(rgb)

            # ----------------------------------------------------------
            # 2A. No target hand
            # ----------------------------------------------------------
            if num_box == 0:
                diagnostics.on_detection_invalid()

                # Recording is independent of UDP and therefore still
                # occurs during Gate 3A --dry-run.
                recorder.write_invalid_detection(
                    seq=seq,
                    timestamp=capture_wall_s,
                    capture_monotonic_s=capture_monotonic_s,
                )

                if not args.dry_run:
                    send_invalid_tracking_packet(
                        sock,
                        args.host,
                        args.port,
                        seq,
                    )
                    diagnostics.on_sent_invalid()

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
                        cv2.waitKey(1) & 0xFF
                    ) == ord("q"):
                        break

                diagnostics.maybe_print_invalid()
                continue

            diagnostics.on_detection_valid()

            # ----------------------------------------------------------
            # 3. Raw human task-space vectors
            #    v_human = task landmark - origin landmark
            # ----------------------------------------------------------
            reference_vectors_raw = (
                joint_pos[task_indices, :]
                - joint_pos[origin_indices, :]
            )
            diagnostics.update_reference_vectors(
                reference_vectors_raw
            )

            # ----------------------------------------------------------
            # 4. Human coordinate frame -> ORCA coordinate frame
            #    V_orca = scale * V_human @ R.T
            # ----------------------------------------------------------
            reference_vectors_aligned = (
                apply_vector_alignment(
                    reference_vectors_raw,
                    alignment_rotation,
                    alignment_scale,
                )
            )

            if not np.all(
                np.isfinite(reference_vectors_aligned)
            ):
                print(
                    "[WARNING] Aligned Human->ORCA reference "
                    "vectors contain NaN/Inf."
                )
                diagnostics.on_retarget_failure()

                if not args.dry_run:
                    send_invalid_tracking_packet(
                        sock,
                        args.host,
                        args.port,
                        seq,
                    )
                    diagnostics.on_sent_invalid()

                seq += 1
                continue

            diagnostics.update_aligned_vectors(
                reference_vectors_aligned
            )

            # ----------------------------------------------------------
            # 5. dex-retargeting VectorOptimizer
            # ----------------------------------------------------------
            start_retarget = time.perf_counter()
            qpos = retargeting.retarget(
                reference_vectors_aligned,
                fixed_qpos=fixed_qpos,
            )
            retarget_ms = (
                time.perf_counter() - start_retarget
            ) * 1000.0

            # Preserve the original latest-time behavior even if an
            # optimizer result becomes invalid.
            diagnostics.latest_retarget_ms = float(
                retarget_ms
            )

            if not np.all(np.isfinite(qpos)):
                diagnostics.on_retarget_failure()
                print(
                    "[WARNING] Retargeting returned NaN/Inf."
                )

                if not args.dry_run:
                    send_invalid_tracking_packet(
                        sock,
                        args.host,
                        args.port,
                        seq,
                    )
                    diagnostics.on_sent_invalid()

                seq += 1
                continue

            diagnostics.on_retarget_success()
            diagnostics.update_qpos(
                qpos,
                retarget_ms,
            )

            # ----------------------------------------------------------
            # 6. Packet construction + Gate 3A JSONL record
            # ----------------------------------------------------------
            packet = build_valid_tracking_packet(
                seq=seq,
                timestamp=capture_wall_s,
                joint_names=joint_names,
                qpos=qpos,
                retarget_ms=retarget_ms,
            )

            recorder.write_valid(
                packet=packet,
                capture_monotonic_s=capture_monotonic_s,
                joint_pos=joint_pos,
                keypoint_2d=keypoint_2d,
                reference_vectors_raw=reference_vectors_raw,
                reference_vectors_aligned=reference_vectors_aligned,
                reference_vector_lengths=(
                    diagnostics.latest_reference_lengths
                ),
            )

            # ----------------------------------------------------------
            # 7. Optional UDP
            # ----------------------------------------------------------
            if not args.dry_run:
                send_packet(
                    sock,
                    args.host,
                    args.port,
                    packet,
                )
                diagnostics.on_sent_valid()

            seq += 1
            diagnostics.maybe_print_valid(seq)

            # ----------------------------------------------------------
            # 8. Optional GUI (unchanged; container Qt/XCB must work)
            # ----------------------------------------------------------
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
                    f"retarget: {retarget_ms:.2f} ms",
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
                    cv2.waitKey(1) & 0xFF
                ) == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        cap.release()
        sock.close()
        recorder.close()

        if args.show:
            cv2.destroyAllWindows()

        print()
        retargeting.verbose()


if __name__ == "__main__":
    main()
