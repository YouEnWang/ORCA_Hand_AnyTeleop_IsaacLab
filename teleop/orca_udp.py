#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UDP packet construction and transmission for ORCA teleoperation.

Purpose
-------
This module owns only network-packet concerns.  It does not perform
camera capture, MediaPipe inference, coordinate alignment, optimization,
or logging.

Separating UDP from the main loop is important for Gate 3A because
``--dry-run`` must disable *all* network transmission while the same
perception/alignment/retargeting path continues to run and be recorded.

Behavior-preservation note
--------------------------
The packet schemas intentionally match the pre-refactor server:

Valid tracking packet::

    {
        "seq": int,
        "timestamp": float,
        "tracking_valid": True,
        "joint_names": [...],
        "positions_rad": [...],
        "retarget_ms": float,
        "source": "mediapipe_dex_retargeting"
    }

Invalid tracking packet::

    {
        "seq": int,
        "timestamp": time.time(),
        "tracking_valid": False,
        "source": "mediapipe"
    }
"""

from __future__ import annotations

import json
import socket
import time
from typing import Sequence

import numpy as np


def build_valid_tracking_packet(
    *,
    seq: int,
    timestamp: float,
    joint_names: Sequence[str],
    qpos: np.ndarray,
    retarget_ms: float,
) -> dict:
    """Build the exact valid-command packet used by the server."""

    return {
        "seq": seq,
        "timestamp": timestamp,
        "tracking_valid": True,
        "joint_names": list(joint_names),
        "positions_rad": (
            np.asarray(qpos)
            .astype(float)
            .tolist()
        ),
        "retarget_ms": float(retarget_ms),
        "source": "mediapipe_dex_retargeting",
    }


def send_packet(
    sock: socket.socket,
    host: str,
    port: int,
    packet: dict,
) -> None:
    """Serialize one dictionary as UTF-8 JSON and transmit it via UDP."""

    sock.sendto(
        json.dumps(packet).encode("utf-8"),
        (host, port),
    )


def send_invalid_tracking_packet(
    sock: socket.socket,
    host: str,
    port: int,
    seq: int,
) -> None:
    """
    Send the historical invalid-tracking packet.

    The timestamp is intentionally generated here with ``time.time()`` to
    preserve the behavior of the pre-refactor implementation.
    """

    packet = {
        "seq": seq,
        "timestamp": time.time(),
        "tracking_valid": False,
        "source": "mediapipe",
    }

    send_packet(
        sock,
        host,
        port,
        packet,
    )
