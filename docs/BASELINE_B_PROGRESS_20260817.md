# Baseline B Progress — AnyTeleop-style ORCA Hand Retargeting

Date: 2026-08-17

## Goal

Integrate an AnyTeleop-style vision-based dexterous hand
retargeting pipeline with ORCA Hand v2 Right in Isaac Lab.

Current pipeline:

```text
RealSense / RGB
    ↓
MediaPipe SingleHandDetector
    ↓
21 × 3 human hand landmarks
    ↓
10 human reference vectors
    ↓
Human-to-ORCA coordinate alignment
    ↓
dex-retargeting VectorOptimizer
    ↓
17-DoF ORCA joint targets
    ↓
UDP
    ↓
Isaac Lab ORCA Hand client
```

## Completed

1. ORCA Hand Import

  - ORCA Hand v2 Right URDF successfully loaded with Pinocchio.
  - ORCA Hand successfully imported into Isaac Sim / Isaac Lab.
  - 17 revolute DoFs are available.

2. UDP Communication
  - Retargeting server sends ORCA joint targets through UDP.
  - Isaac Lab client receives packets at approximately 30 Hz.
  - Packet-to-Isaac joint-name mapping was verified.
  - Isaac desired, applied, and actual joint positions track correctly.

3. Human Reference Vector Diagnostics

  The AnyTeleop-style human vector representation uses 10 vectors:

  - Wrist → five fingertips
  - Wrist → five intermediate finger landmarks

  Open → Fist → Open experiments confirmed that the human reference vectors change consistently with finger motion.

4. ORCA Robot Reference Vector Diagnostics

  Pinocchio FK was used to calculate the 10 ORCA robot vectors at q = 0.

  A Middle/Ring task-link mapping error was identified and fixed.

5. Human-to-ORCA Coordinate Alignment

  A global Human → ORCA coordinate-frame alignment was estimated using non-thumb vectors.

  The calibration solves:

  ```text
  V_orca = scale × V_human × R^T
  ```

  Current calibrated scale:

  ```text
  scale ≈ 0.752138
  ```

  The resulting rotation matrix has:

  ```text
  det(R) ≈ 1.0
  ```

  The non-thumb calibration RMSE is approximately:

  ```text
  12.9 mm
  ```
  
6. Realtime ORCA Retargeting

  After applying the Human → ORCA alignment before the VectorOptimizer:

  - Index, Middle, Ring, and Pinky begin to respond clearly to human Open → Fist → Open motion.
  - The previous degenerate solution in which most joints remained fixed at their limits was substantially resolved.
  - Isaac Lab visually confirms large finger opening/closing motion.

## Remaining Issues

### PIP joints

The following ORCA joints currently remain at approximately zero:

  - Index PIP
  - Middle PIP
  - Ring PIP
  - Pinky PIP
  - Thumb distal joint

The current fingertip task frames are likely located close to the
corresponding joint axes, making these DoFs weakly observable in
the position-based VectorOptimizer objective.

### Thumb

Thumb retargeting remains inaccurate because the ORCA thumb
morphology differs significantly from the other four fingers.

### Abduction joints

ORCA Hand v2 has an active abduction/adduction motor at the base
of each non-thumb finger.

Future retargeting must preserve and correctly map:

```text
ABD lateral motion
+
MCP flexion
+
PIP flexion
```

rather than treating the fingers as flexion-only chains.

## Next Step

Step 7:

1. Verify the complete ORCA Hand v2 joint semantics.
2. Verify positive/negative direction of each ABD/MCP/PIP joint.
3. Define proper fingertip virtual retargeting frames.
4. Make PIP motion observable to VectorOptimizer.
5. Re-run Open/Fist/Open and finger-spreading tests.
6. Improve thumb retargeting separately.
