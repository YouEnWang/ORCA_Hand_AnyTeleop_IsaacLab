# ORCA Hand v2 Teleoperation Progress — 2026-08-21

## Current Goal

Use camera-based human hand pose estimation to control ORCA Hand v2 with 16 finger DoF from `dex-retargeting` `VectorOptimizer` and 1 wrist DoF from a separate wrist retargeter.

## Current Architecture

```text
Camera / MediaPipe
        |
        +-- Local finger pose --> VectorOptimizer --> 16 finger DoF
        |
        +-- Wrist orientation --> Wrist Retargeter --> 1 wrist DoF
                                      |
                                      v
                                  17-DoF command
                                      |
                                      v
                                     UDP
                                      |
                                      v
                               Isaac Lab Client
                                      |
                                      v
                                ORCA Hand v2
```

## Completed

- ORCA Hand v2 17-DoF joint semantics verified.
- Wrist is excluded from `VectorOptimizer`; finger retargeting optimizes 16 DoF.
- Virtual fingertip frames make non-thumb PIP joints observable.
- Human-to-ORCA vector alignment calibrated.
- Real-time finger open/fist/open teleoperation works in Isaac Lab.
- UDP packet-to-Isaac joint mapping verified.
- Separate 1-DoF wrist retargeter added.
- Swing-twist decomposition is used for ORCA wrist-axis rotation.
- Wrist command is merged with the 16 finger DoF after `VectorOptimizer`.
- Current wrist mapping:
  - `sign: -1.0`
  - `safe_lower_rad: -0.15`
  - `safe_upper_rad: 0.15`
- Bidirectional wrist commands were confirmed.
- Combined 17-DoF control path is working.

## Current Issues

### Isaac wrist oscillation

The wrist visibly oscillates in Isaac Sim even when the target is relatively stable.

Planned follow-up:
- log wrist-specific desired/applied/actual/error/velocity;
- tune wrist stiffness and damping separately.

### MediaPipe robustness during wrist rotation

Some wrist orientations noticeably distort the estimated finger geometry even while the real hand remains open. This can push finger retargeting outputs toward joint limits.

## Current Status

- 16-DoF finger retargeting: PASS
- 1-DoF wrist retargeting: PASS
- Bidirectional wrist command: PASS
- 17-DoF integration: PASS
- UDP communication: PASS
- Isaac wrist stability: NOT YET RESOLVED
- Physical ORCA Hand v2 deployment: NEXT MAJOR STEP

## Next Major Step

Replace the Isaac Lab client with a hardware client:

```text
Camera / MediaPipe
        |
        v
Finger + Wrist Retargeting
        |
        v
17-DoF ORCA command
        |
        v
ORCA Hardware Client
        |
        v
orca_core
        |
        v
Physical ORCA Hand v2
```

Before full real-time teleoperation, complete the physical hand's normal tensioning, calibration, neutral-position verification, joint-limit checks, and low-speed command tests.
