# ORCA Hand v2 Vision-Based Teleoperation in Isaac Lab

本專案旨在建立一套以 **AnyTeleop** 為概念參考的 ORCA Hand v2 裸手視覺即時遙操作系統。

目前的研究基礎為：

- 使用 Intel RealSense D435i 擷取人手影像
- 使用 MediaPipe 偵測 21 個人手關鍵點
- 根據人手骨架計算手指關節角度
- 使用 joint-angle mapping 即時控制 Isaac Lab 中的 Shadow Hand
- 記錄並分析 human joint angles、robot targets、Isaac applied targets 與 actual joint positions
- 使用濾波與 Daubechies db4 小波分析關節抖動

下一階段將把目前的 Shadow Hand pipeline 移植至實驗室的 **ORCA Hand v2 Right**，並逐步建立：

1. ORCA Hand v2 在 Isaac Lab 中的模擬模型
2. MediaPipe 至 ORCA Hand 的直接關節角映射
3. AnyTeleop-inspired optimization-based retargeting
4. 模擬與實體 ORCA Hand 的統一控制介面
5. 視覺估測誤差、重定向誤差、延遲與關節抖動分析

---

## 1. Research Objective

本專案目前的研究目標為：

> 建立一套適用於 ORCA Hand v2 的完整 vision-based dexterous hand teleoperation pipeline，並比較直接關節角映射與 optimization-based retargeting 在控制精度、指尖位置、關節抖動與系統延遲上的差異。

目前研究範圍聚焦於單一靈巧手，不包含：

- 機械手臂控制
- CuRobo motion generation
- 多相機融合
- Web-based remote visualizer
- 多操作者或多機器人協作

因此，本專案現階段定位為：

> **AnyTeleop-inspired dexterous hand teleoperation for ORCA Hand v2**

而不是完整重建 AnyTeleop 的 arm-hand teleoperation system。

---

## 2. Target System

### Hardware

- Dexterous hand: ORCA Hand v2 Right
- Degrees of freedom: 17 DoF
- Motors:
  - FeeTech HL2915
  - FeeTech HL3930
- Camera: Intel RealSense D435i
- Host operating system: Linux

### Software

- Isaac Sim
- Isaac Lab
- Python
- MediaPipe
- Intel RealSense SDK / `pyrealsense2`
- NumPy
- OpenCV
- PyYAML
- Git

Future integration may additionally use:

- `orca_core`
- `dex-retargeting`
- Pinocchio
- SciPy optimization tools

---

## 3. Planned Teleoperation Pipeline

```text
Intel RealSense D435i
        │
        ▼
RGB / Aligned Depth Images
        │
        ▼
MediaPipe Hand Detection
        │
        ├── 21×2 image landmarks
        ├── 21×3 hand landmarks
        └── hand tracking status
        │
        ▼
Coordinate and Hand-Frame Processing
        │
        ├── wrist-centered coordinates
        ├── scale normalization
        ├── depth-based 3D reconstruction
        └── temporal validity checking
        │
        ▼
Human Hand Representation
        │
        ├── human joint angles
        └── task-space key vectors
        │
        ▼
Human-to-ORCA Retargeting
        │
        ├── Direct joint-angle mapping
        ├── AnyTeleop-style optimization
        └── Proposed robust retargeting
        │
        ▼
Command Safety and Filtering
        │
        ├── joint limits
        ├── velocity limits
        ├── command timeout
        ├── tracking-loss handling
        └── optional jitter suppression
        │
        ├──────────────────────────┐
        ▼                          ▼
Isaac Lab ORCA Hand       Physical ORCA Hand v2
        │                          │
        ▼                          ▼
Applied / Actual State    Motor / Joint Feedback
        └────────── Logging ───────┘
```

---

## 4. Current Progress

### Completed

* [x] RealSense D435i RGB-D image acquisition
* [x] MediaPipe 21-landmark hand tracking
* [x] Human finger joint-angle calculation
* [x] Direct joint-angle mapping for Shadow Hand
* [x] Real-time UDP communication with Isaac Lab
* [x] Shadow Hand real-time control in Isaac Lab
* [x] Robot raw target and filtered target logging
* [x] Isaac applied target logging
* [x] Isaac actual joint position logging
* [x] Timestamp and communication-delay logging
* [x] Daubechies db4 wavelet jitter analysis
* [x] Preliminary filtering experiments

### In Progress

* [ ] Clone and inspect the official ORCA Hand description repository
* [ ] Audit the ORCA Hand v2 Right URDF
* [ ] Resolve URDF mesh paths
* [ ] Convert ORCA Hand v2 URDF to USD
* [ ] Import ORCA Hand v2 into Isaac Sim
* [ ] Verify the 17 ORCA joints in Isaac Lab
* [ ] Build an ORCA semantic joint mapping table
* [ ] Implement individual joint sweep tests

### Future Work

* [ ] Implement MediaPipe-to-ORCA direct joint mapping
* [ ] Integrate AnyTeleop-style optimization-based retargeting
* [ ] Compare joint mapping and task-space retargeting
* [ ] Integrate the physical ORCA Hand through `orca_core`
* [ ] Compare simulated and physical ORCA joint responses
* [ ] Design uncertainty-aware and jitter-aware retargeting
* [ ] Perform basic grasping and pinching experiments

---

## 5. Repository Structure

```text
orca_isaaclab_import/
├── README.md
├── THIRD_PARTY_VERSIONS.md
├── .gitignore
│
├── external/
│   └── orcahand_description/
│       └── Official ORCA Hand description repository
│
├── assets/
│   ├── urdf_work/
│   │   └── Modified URDF copies for Isaac Sim
│   └── usd/
│       └── Generated Isaac Sim USD assets
│
├── config/
│   ├── orca_v2_joint_bridge.yaml
│   ├── orca_v2_actuators.yaml
│   ├── orca_direct_mapping.yaml
│   └── orca_anyteleop_retargeting.yaml
│
├── tools/
│   ├── audit_orca_v2_urdf.py
│   ├── convert_orca_urdf.py
│   ├── inspect_orca_usd.py
│   ├── sweep_isaac_orca_joints.py
│   └── compare_orca_joint_names.py
│
├── isaaclab/
│   ├── orca_hand_cfg.py
│   ├── run_orca_hand.py
│   └── orca_udp_receiver.py
│
├── hpe/
│   ├── mediapipe_source.py
│   ├── depth3d.py
│   └── hand_frame.py
│
├── retargeting/
│   ├── direct_mapping.py
│   ├── anyteleop_optimizer.py
│   └── robust_optimizer.py
│
├── safety/
│   ├── command_limiter.py
│   └── watchdog.py
│
├── recording/
│   └── full_recorder.py
│
├── analysis/
│   ├── joint_response_analysis.py
│   ├── jitter_analysis.py
│   ├── latency_analysis.py
│   └── sim_real_comparison.py
│
└── logs/
    └── Generated experiment logs
```

Not all directories and files have been implemented yet. The structure above represents the planned modular architecture.

---

## 6. Third-Party ORCA Hand Description

The official ORCA Hand model is obtained from:

```text
https://github.com/orcahand/orcahand_description
```

Clone the repository into the `external/` directory:

```bash
mkdir -p ~/Projects/orca_isaaclab_import/external

cd ~/Projects/orca_isaaclab_import/external

git clone \
  https://github.com/orcahand/orcahand_description.git
```

The target model is:

```text
ORCA Hand v2 Right
```

The expected URDF is located under the `v2` directory. Confirm its actual path with:

```bash
find \
  ~/Projects/orca_isaaclab_import/external/orcahand_description/v2 \
  -type f \
  \( -iname "*.urdf" -o -iname "*.xacro" \) \
  | sort
```

The entire repository should be retained because the URDF references external mesh assets. Downloading only the `.urdf` file may result in missing visual meshes.

---

## 7. Environment Variables

The following environment variables can be used for convenience:

```bash
export ORCA_PROJECT=~/Projects/orca_isaaclab_import

export ORCA_DESC=\
$ORCA_PROJECT/external/orcahand_description

export ORCA_URDF=\
$ORCA_DESC/v2/models/urdf/orcahand_right.urdf
```

Check whether the URDF exists:

```bash
test -f "$ORCA_URDF" \
  && echo "[OK] ORCA URDF found: $ORCA_URDF" \
  || echo "[ERROR] ORCA URDF not found"
```

Check whether the URDF contains ROS package paths:

```bash
grep -n "package://" "$ORCA_URDF" | head -20
```

Check available mesh assets:

```bash
find "$ORCA_DESC/v2" \
  -type f \
  \( -iname "*.stl" \
     -o -iname "*.obj" \
     -o -iname "*.dae" \) \
  | head -30
```

---

## 8. Isaac Lab URDF Conversion

The exact URDF conversion script path depends on the installed Isaac Lab version.

Locate the conversion script:

```bash
cd ~/IsaacLab

find . -type f -name "convert_urdf.py"
```

Inspect the available arguments:

```bash
./isaaclab.sh -p \
  <PATH_TO_CONVERT_URDF_PY> \
  --help
```

The first ORCA conversion should use the following principles:

* Use the ORCA Hand v2 Right URDF
* Fix the base of the hand
* Do not connect MediaPipe during the first import
* Do not immediately trust the imported actuator parameters
* Do not initially merge all fixed joints
* Verify mesh paths before conversion
* Preserve the original URDF and modify only a copied version

If a working URDF copy is required:

```bash
cp "$ORCA_URDF" \
  "$ORCA_PROJECT/assets/urdf_work/orcahand_right_isaac.urdf"
```

All Isaac-specific changes should be made to this copied file rather than the official file inside `external/orcahand_description`.

---

## 9. ORCA Joint Audit

Finding the official URDF does not eliminate the need for joint auditing.

The ORCA URDF describes:

* Links
* Kinematic joints
* Joint axes
* Joint limits
* Visual meshes
* Mass and inertia information

However, it does not fully define:

* ORCA semantic joint names
* FeeTech motor IDs
* HL2915 / HL3930 assignments
* Motor inversion
* Hardware neutral positions
* Tendon calibration
* Physical joint ROM
* Isaac joint ordering
* Sim-to-real direction consistency

The project therefore requires alignment among four representations:

```text
MediaPipe human-hand representation
        ↕
ORCA URDF kinematic joints
        ↕
Isaac Lab articulation joints
        ↕
orca_core semantic joints and FeeTech motors
```

The final joint bridge is expected to be stored in:

```text
config/orca_v2_joint_bridge.yaml
```

Example structure:

```yaml
joints:
  - semantic_name: wrist
    urdf_name: null
    isaac_name: null
    isaac_index: null
    motor_id: null
    motor_type: HL3930
    hardware_inverted: false
    lower_rad: null
    upper_rad: null
    neutral_rad: null

  - semantic_name: thumb_cmc
    urdf_name: null
    isaac_name: null
    isaac_index: null
    motor_id: null
    motor_type: null
    hardware_inverted: null
    lower_rad: null
    upper_rad: null
    neutral_rad: null
```

The values should be filled only after inspecting the URDF, Isaac articulation and physical hardware.

---

## 10. First Validation Milestone

The first milestone is complete only when all of the following conditions are satisfied:

* [ ] ORCA Hand v2 Right URDF can be parsed
* [ ] All required mesh files can be resolved
* [ ] The URDF can be converted to USD
* [ ] The ORCA Hand model appears correctly in Isaac Sim
* [ ] The base is fixed correctly
* [ ] All expected controllable joints are present
* [ ] Joint limits are available
* [ ] Each joint can be moved individually
* [ ] The positive and negative joint directions are recorded
* [ ] The Isaac joint names and indices are exported
* [ ] A preliminary semantic joint mapping is completed

At this stage, MediaPipe and real-time teleoperation should remain disabled.

---

## 11. Planned Retargeting Methods

Three retargeting methods are planned.

### Method A: Direct Joint-Angle Mapping

```text
Human joint angles
        ↓
Scale and offset
        ↓
ORCA joint targets
```

This method will serve as the baseline.

### Method B: AnyTeleop-Style Optimization

The robot joint state will be optimized according to human and robot task-space vectors:

```math
q_t^*
=
\arg\min_q
\left[
\sum_i
w_i
\left\|
\alpha v_i^{human}
-
v_i^{robot}(q)
\right\|^2
+
\beta
\left\|
q-q_{t-1}
\right\|^2
\right]
```

subject to:

```math
q_{\min}\le q\le q_{\max}
```

The vectors may include:

* Palm-to-fingertip vectors
* Finger-base-to-fingertip vectors
* Thumb-to-index pinch vector
* Thumb-to-middle pinch vector
* Finger direction vectors
* Finger-abduction relationships

### Method C: Proposed Robust Retargeting

The proposed method may additionally use:

* Landmark validity
* Depth validity
* Temporal consistency
* Huber loss
* Adaptive temporal regularization
* Per-joint confidence weights
* Jitter-aware filtering
* Tendon-driven response compensation

---

## 12. Planned Evaluation

### Pose Tracking

* Open hand
* Closed fist
* Individual finger flexion
* Finger abduction
* Thumb opposition
* Thumb-index pinch
* Three-finger pinch

### Static Jitter

The user maintains a fixed gesture for 20–30 seconds.

Analysis will include:

* Landmark standard deviation
* Human-angle jitter
* Retargeting-target jitter
* Isaac actual-joint jitter
* RMS
* Power spectral density
* Daubechies db4 wavelet coefficients

### Dynamic Response

* Open hand to fist
* Open hand to pinch
* Slow and fast finger flexion

Metrics:

* End-to-end latency
* Rise time
* Settling time
* Overshoot
* Trajectory RMSE
* Phase delay

### Basic Manipulation

Future task examples:

* Cylindrical power grasp
* Cube grasp
* Small-object pinch
* Cup grasp
* Object pick-and-place

---

## 13. Git and Dependency Policy

The official ORCA repository is treated as an external dependency:

```text
external/orcahand_description/
```

It is not directly included in this repository's Git history.

The exact third-party version is recorded in:

```text
THIRD_PARTY_VERSIONS.md
```

Files expected to be committed:

* Source code
* Configuration files
* Audit tools
* Modified URDF copies
* Isaac Lab asset configuration
* Documentation
* Analysis scripts
* Small reproducible examples

Files normally excluded:

* External repositories
* Generated USD files
* Isaac and Omniverse caches
* Virtual environments
* Large CSV logs
* Videos
* ROS bags
* Temporary outputs

---

## 14. Safety Notice

The current development stage focuses on simulation.

Before connecting the physical ORCA Hand, the following must be implemented and verified:

* Joint-limit enforcement
* Velocity limits
* Command timeout
* Tracking-loss watchdog
* NaN and invalid-command rejection
* Motor-current monitoring
* Motor-temperature monitoring
* Emergency stop
* Safe neutral or open-hand behavior
* Tendon calibration and tension verification

MediaPipe or optimizer output must not be sent directly to the physical ORCA Hand without a safety layer.

---

## 15. Research Direction

The current proposed thesis direction is:

> Robust vision-based teleoperation of the ORCA Hand v2 using uncertainty-aware retargeting and joint-jitter suppression.

A possible Chinese title is:

> 結合不確定性感知重定向與關節抖動抑制之 ORCA Hand v2 裸手視覺即時遙操作系統

A possible English title is:

> An AnyTeleop-Inspired Robust Vision-Based Teleoperation Pipeline for the ORCA Hand v2

---

## 16. References

The project is conceptually inspired by:

* DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System
* AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System

Official ORCA resources:

* ORCA Hand description:
  `https://github.com/orcahand/orcahand_description`
* ORCA Hand core:
  `https://github.com/orcahand/orca_core`

Additional references and exact dependency versions are recorded in:

```text
THIRD_PARTY_VERSIONS.md
```

---

## 17. Project Status

The project is currently at the **ORCA Hand v2 URDF import and model-audit stage**.

The next immediate tasks are:

1. Clone the official ORCA Hand description repository
2. Locate and inspect the ORCA Hand v2 Right URDF
3. Resolve all mesh paths
4. Audit URDF joints, limits and geometry
5. Convert the URDF into USD
6. Import the model into Isaac Sim
7. Verify every controllable joint through individual joint sweeps



另外，你可以在 README 最前面或最後面加入一段目前實際使用版本，等你確認後再填入：

```markdown
## Development Environment

- Operating system: Ubuntu XX.XX
- GPU: TBD
- NVIDIA driver: TBD
- CUDA: TBD
- Isaac Sim: TBD
- Isaac Lab branch: TBD
- Isaac Lab commit: TBD
- Python: TBD
- ORCA description commit: TBD


這些內容之後應與 `THIRD_PARTY_VERSIONS.md` 保持一致。
