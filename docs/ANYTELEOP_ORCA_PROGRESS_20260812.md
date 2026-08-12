# AnyTeleop–ORCA Hand v2 Integration Progress

Date: 2026-08-12

## 1. Objective

The current objective is to build an AnyTeleop-inspired
vision-based dexterous-hand teleoperation pipeline for the
ORCA Hand v2 Right.

The current target architecture is:

```text
D435i RGB Camera
        |
        v
MediaPipe Hand Detection
        |
        | 21 x 3 hand landmarks
        v
dex-retargeting
VectorOptimizer
        |
        | ORCA joint names + qpos
        v
UDP Communication
        |
        v
Isaac Lab Teleoperation Client
        |
        v
ORCA Hand v2 Right
in Isaac Sim
```

The current stage focuses only on the dexterous hand.

The following AnyTeleop components are not yet integrated:

* Robot arm motion generation
* Multi-camera detection fusion
* CuRobo
* Web visualizer
* Physical ORCA Hand
* Remote multi-user operation

---

## 2. ORCA Hand Simulation Model

The official ORCA Hand v2 Right URDF was obtained from:

```text
https://github.com/orcahand/orcahand_description
```

ORCA description commit:

```text
b9b349a21ee0238c62b6cf92ae7597027867adf8
```

The official URDF was converted into an Isaac Sim USD model.

The converted ORCA model has been successfully displayed in
Isaac Sim 5.1.0.

Formal USD inspection confirmed:

```text
Articulation root : present
Revolute joints   : 17
Prismatic joints  : 0
Fixed base        : enabled
```

The current model is suitable for initial pose-retargeting tests.

Collision geometry has not yet been validated for physical grasping
experiments.

---

## 3. URDF Import Compatibility Fix

The original CAD-exported ORCA assets contain hyphens in several
mesh filenames.

These filenames caused invalid USD prim paths during the original
URDF conversion and resulted in:

```text
Ill-formed SdfPath
RuntimeError: Used null prim
```

A sanitization tool was created:

```text
tools/sanitize_orca_urdf_assets.py
```

The tool creates an Isaac-compatible working copy while preserving
the official ORCA repository unchanged.

After sanitization, the ORCA Hand v2 Right model can be successfully
converted to USD and opened in Isaac Sim.

---

## 4. AnyTeleop Retargeting Environment

The retargeting system is based on:

```text
https://github.com/dexsuite/dex-retargeting
```

Pinned commit:

```text
3f56141bc8bd2760d5e452e382937269554ebb21
```

The retargeting module is executed inside an independent Docker
container.

This follows the modular/containerized design philosophy described
by AnyTeleop and also prevents dependency conflicts with the
Isaac Lab runtime.

Current architecture:

```text
Docker Container
----------------
MediaPipe
dex-retargeting
Pinocchio
PyTorch CPU
OpenCV
UDP Publisher

        |
        | UDP
        v

Host Linux
----------
Isaac Lab
Isaac Sim
ORCA Hand v2
UDP Client
```

---

## 5. Container Dependency Issues Resolved

### PyTorch

The pinned dex-retargeting implementation imports PyTorch.

A CPU-only PyTorch installation was therefore added to the
retargeting container.

GPU execution is currently unnecessary for the retargeting module.

### MediaPipe

A newer MediaPipe package did not provide:

```python
mediapipe.framework
```

which is used by the pinned dex-retargeting
`SingleHandDetector`.

The environment was therefore pinned to:

```text
MediaPipe 0.10.21
```

to maintain compatibility with the AnyTeleop/dex-retargeting
example code.

---

## 6. RealSense Camera Access

The Intel RealSense D435i video device has been exposed to the
retargeting Docker container.

Example Docker device mapping:

```text
/dev/video4
```

OpenCV successfully opened and read the video stream:

```text
Opened: True
Read: True
```

The exact RealSense RGB video node still needs to be formally
verified before final experiments.

---

## 7. ORCA Vector Retargeting

The current retargeting method is:

```text
VectorOptimizer
```

Instead of directly mapping human joint angles to robot joint angles,
the optimizer compares task-space vectors between the human hand
and the ORCA Hand.

For a robot vector:

```text
robot_vector =
    task_link_position
    -
    origin_link_position
```

The current initial configuration uses vectors such as:

```text
ORCA palm -> thumb tip
ORCA palm -> index tip
ORCA palm -> middle tip
ORCA palm -> ring tip
ORCA palm -> pinky tip
```

and corresponding intermediate finger vectors.

The human-side references are derived from MediaPipe hand landmarks.

The current ORCA-specific retargeting configuration is:

```text
config/retargeting/orca_v2_right_vector.yml
```

The ORCA wrist is currently fixed, while the finger joints are
optimized.

---

## 8. Network Communication

The retargeting server and Isaac Lab client communicate through UDP.

Current port:

```text
5006
```

The packet contains:

```text
sequence number
timestamp
tracking validity
robot joint names
robot joint positions
```

Joint names are transmitted together with qpos because the joint
ordering used by dex-retargeting is different from the Isaac Lab
runtime joint ordering.

---

## 9. Joint-Name Mapping

The first complete AnyTeleop-to-Isaac communication test successfully
constructed a mapping between all 17 dex-retargeting joint names and
all 17 Isaac Lab runtime joints.

Example:

```text
dex-retargeting:
I-PP_bacbd481_to_I-AP-R_d95d02d1

        |
        v

Isaac:
I_PP_bacbd481_to_I_AP_R_d95d02d1
```

The difference is primarily caused by Isaac Sim sanitizing unsupported
USD-name characters such as hyphens into underscores.

The full 17-joint mapping was successfully constructed without missing
joints.

This confirms that:

```text
dex-retargeting
        |
        v
UDP
        |
        v
Isaac joint-name mapping
```

is operational.

---

## 10. Isaac Lab Teleoperation Client

The current Isaac Lab client is:

```text
isaaclab/orca_anyteleop_client.py
```

Its responsibilities are:

```text
Receive UDP packet
        |
        v
Validate packet
        |
        v
Joint-name mapping
        |
        v
Joint-limit clamp
        |
        v
Command slew-rate limiting
        |
        v
Isaac position target
```

The Isaac client does not run MediaPipe or dex-retargeting.

This separation is intentional so that the same future retargeting
server can later be connected to a physical ORCA Hand client.

---

## 11. Viewport Issue

An apparent Isaac Sim camera-reset problem was investigated.

The issue was not caused by continuous calls to:

```python
sim.set_camera_view()
```

The ORCA Hand is a fixed-base articulation.

Moving `/World/ORCAHand` manually while physics is running modifies
the robot transform rather than the Perspective camera. PhysX then
maintains the fixed-base articulation at its physical pose, which
appears as if the view is snapping back.

The issue is resolved by manipulating the Perspective viewport instead
of moving the ORCA articulation.

Current recommended viewport controls include:

```text
F              Focus selected ORCA Hand
Alt + LMB      Orbit
Middle Mouse   Pan
Mouse Wheel    Zoom
```

---

## 12. Current Full-Pipeline Test

The following pipeline has been executed:

```text
D435i
  |
  v
MediaPipe / SingleHandDetector
  |
  v
VectorOptimizer
  |
  v
ORCA qpos
  |
  v
UDP
  |
  v
Isaac Lab Client
  |
  v
ORCA Hand v2 Right
```

The first valid packet successfully reached Isaac Lab.

The complete 17-joint packet-to-Isaac mapping was constructed.

The ORCA Hand responded to the first control target.

Therefore, the following modules have been connected successfully:

```text
Camera
MediaPipe
dex-retargeting
UDP
Isaac Lab
ORCA articulation
```

---

## 13. Current Known Issue

The current full-pipeline behavior is not yet correct.

Observed behavior:

```text
First valid retargeting command
        |
        v
ORCA Hand quickly flexes / contracts
        |
        v
The hand then appears to stop following human motion
```

This issue has NOT yet been fixed in the current checkpoint.

Possible causes identified for future debugging include:

1. dex-retargeting initial qpos
2. initial VectorOptimizer solution
3. MediaPipe tracking validity after the first frame
4. handedness filtering
5. camera stream selection
6. retargeting scaling
7. retargeting low-pass behavior
8. command update frequency

One important implementation detail identified is that
`SeqRetargeting` initializes previous qpos using values related to
joint-limit ranges rather than a calibrated ORCA open-hand pose.

The optimizer initialization has intentionally NOT yet been modified
in this checkpoint.

---

## 14. Next Steps

The next debugging stage will focus on the dynamic retargeting path.

Planned checks:

```text
1. Add MediaPipe valid/invalid detection statistics
2. Add UDP receive-rate monitoring
3. Add valid/invalid packet counters
4. Log consecutive qpos values
5. Verify the D435i RGB video node
6. Inspect initial VectorOptimizer qpos
7. Initialize retargeting from an ORCA open-hand pose
8. Temporarily disable filtering
9. Verify continuous human-to-ORCA motion
```

Only after stable simulated teleoperation is achieved will the
pipeline be extended toward the physical ORCA Hand.

---

## 15. Current Milestone

Current milestone:

> The basic AnyTeleop-style ORCA Hand teleoperation architecture has
> been established, and the first end-to-end command has successfully
> reached the ORCA Hand v2 Right model in Isaac Sim.

Current status:

```text
ORCA URDF / USD integration      PASS
ORCA 17-DoF articulation        PASS
dex-retargeting container       PASS
MediaPipe compatibility         PASS
D435i container access          PASS
VectorOptimizer configuration   INITIAL VERSION
UDP communication               PASS
17-joint name mapping           PASS
Isaac ORCA command reception    PASS
Continuous retargeting          NOT YET VALIDATED
Optimizer initialization        NOT YET FIXED
Physical ORCA Hand              NOT STARTED
Web visualizer                  NOT STARTED
```
