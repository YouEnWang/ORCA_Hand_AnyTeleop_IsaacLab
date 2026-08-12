FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt


# ============================================================
# Pin dex-retargeting
# ============================================================

ARG DEX_RETARGETING_COMMIT

RUN git clone \
    https://github.com/dexsuite/dex-retargeting.git \
    /opt/dex-retargeting \
    && cd /opt/dex-retargeting \
    && git checkout ${DEX_RETARGETING_COMMIT}


# ============================================================
# Python base packages
# ============================================================

RUN python -m pip install --upgrade pip


# ============================================================
# PyTorch
#
# CPU-only Torch is sufficient for dex-retargeting.
# Isaac Sim GPU computation remains on the host.
# ============================================================

RUN python -m pip install \
    torch \
    --index-url https://download.pytorch.org/whl/cpu


# ============================================================
# dex-retargeting
# ============================================================

RUN python -m pip install \
    -e /opt/dex-retargeting


# ============================================================
# MediaPipe compatibility
#
# IMPORTANT:
#
# dex-retargeting's SingleHandDetector uses the legacy
# mediapipe.framework / Solutions API.
#
# MediaPipe >= 0.10.30 removed this API.
#
# Therefore we pin MediaPipe to 0.10.21.
# ============================================================

RUN python -m pip install \
    "mediapipe==0.10.21"


# ============================================================
# dex-retargeting v0.5 uses NumPy >= 2.
#
# MediaPipe 0.10.21 metadata requests NumPy < 2, but
# dex-retargeting explicitly documents compatibility with
# NumPy >= 2 for its MediaPipe examples.
# ============================================================

RUN python -m pip install --upgrade \
    "numpy==2.2.6"


# ============================================================
# Additional packages for our AnyTeleop-style server
# ============================================================

RUN python -m pip install \
    pyzmq \
    pyyaml


# ============================================================
# Build-time environment sanity check
# ============================================================

RUN python - <<'PY'
import sys
import numpy as np
import torch
import cv2
import mediapipe as mp
import mediapipe.framework
import dex_retargeting

print("=" * 70)
print("AnyTeleop Retargeting Container")
print("=" * 70)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("dex_retargeting:", dex_retargeting.__file__)

assert mp.__version__ == "0.10.21"

print("[PASS] mediapipe.framework import succeeded")
print("[PASS] dex-retargeting import succeeded")
PY


# ============================================================
# Verify AnyTeleop / dex-retargeting example detector
# ============================================================

RUN python - <<'PY'
import sys

sys.path.insert(
    0,
    "/opt/dex-retargeting/example/vector_retargeting"
)

from single_hand_detector import SingleHandDetector

print(
    "[PASS] SingleHandDetector import succeeded "
    "during Docker build"
)
PY


# ============================================================
# Project workspace
# ============================================================

WORKDIR /workspace/project

CMD ["bash"]