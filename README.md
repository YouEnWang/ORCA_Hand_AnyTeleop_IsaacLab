# ORCA Hand v2 AnyTeleop-Style Teleoperation Research

本專案旨在建立一套以 **AnyTeleop hand-retargeting pipeline** 為基準的
ORCA Hand v2 Right 裸手視覺即時遙操作系統，並在復現 baseline 的過程中
分析其在低成本、tendon-driven 靈巧手與工業示教任務中的不穩定來源。

目前專案的定位是：

> 復現並移植 AnyTeleop 的 hand-only vision-based retargeting baseline 到
> ORCA Hand v2，先在 Isaac Lab 中驗證，再延伸到實體 ORCA Hand，最後從
> perception uncertainty、retargeting error、command jitter 與 robot response
> mismatch 中收斂碩士論文研究問題。

本專案 **不是完整重建 AnyTeleop 的 arm-hand system**。目前範圍聚焦於單一
ORCA Hand v2 Right，不包含 robot arm motion generation、CuRobo、多相機融合、
web visualizer 或多使用者遠端操作。

---

## 1. Research Direction

建議的碩論方向為：

> 面向 ORCA Hand v2 工業示教任務之不確定性感知重定向與示範資料品質提升

可能的英文題目：

> Uncertainty-Aware Retargeting and Demonstration Quality Enhancement for
> Industrial Teaching with the ORCA Hand v2

研究主軸不是單純展示「ORCA Hand 可以被 MediaPipe 控制」，而是回答：

> 當 AnyTeleop-style vision-based retargeting 被移植到低成本 tendon-driven
> ORCA Hand v2 時，哪些 perception、retargeting 與 robot-response 問題會影響
> 工業示教品質？如何量化並改善這些問題？

預期貢獻包含：

1. 建立 ORCA Hand v2 的 AnyTeleop-style hand-only teleoperation baseline。
2. 在 Isaac Lab 與實體 ORCA Hand 上記錄一致的 command / actual / perception 訊號。
3. 分析 perception uncertainty、retargeting error、command jitter 與 joint response mismatch。
4. 提出 uncertainty-aware retargeting 或 demonstration purification 方法。
5. 以靜態手勢、動態手勢與工業類任務驗證 baseline 與 proposed method 的差異。

---

## 2. Current Status

目前進度大約位於：

> **Gate 2 — ORCA Isaac UDP validation and joint-response characterization 的前半段**

已完成：

- ORCA Hand v2 Right 官方 URDF 取得與 Isaac-compatible 工作副本建立。
- 修正 CAD 匯出 mesh filename 造成的 USD prim path 問題。
- ORCA Hand v2 Right 成功匯入 Isaac Sim / Isaac Lab。
- 確認 ORCA articulation root 與 17 個 revolute joints。
- 建立 containerized dex-retargeting environment。
- pin AnyTeleop-derived `dex-retargeting` implementation。
- 解決 PyTorch 與 MediaPipe compatibility 問題。
- D435i camera device 可從 retargeting container 存取。
- 建立 ORCA-specific VectorOptimizer configuration。
- 建立 AnyTeleop-style UDP retargeting server。
- 建立 Isaac Lab ORCA teleoperation client。
- 完成 dex-retargeting joint names 到 Isaac Lab joint names 的 17-joint mapping。
- 第一筆 retargeted ORCA command 已成功送達 Isaac 並驅動 ORCA model。
- 建立 offline demonstration quality analysis / purification framework。

目前主要未完成或待驗證：

- continuous MediaPipe tracking 與 continuous UDP command transmission。
- ORCA hand 初始快速收縮問題。
- ORCA open-hand optimizer initialization。
- VectorOptimizer scale / alignment calibration。
- 全部手指 retargeted motion 的方向與幅度驗證。
- Gate 2 後半段的 joint-response characterization。
- 實體 ORCA Hand v2 的 safety bridge 與 `orca_core` integration。

---

## 3. Research Gates

以下 gates 是目前建議的研究與開發路線。Gate 3 中明確加入
「復現 AnyTeleop baseline」階段，讓後續研究問題來自 baseline reproduction
時觀察到的實際限制。

### Gate 0 — Research / Repo / Environment Freeze

目標：固定研究問題、第三方版本、baseline、資料格式與實驗設定。

輸出：

- `THIRD_PARTY_VERSIONS.md`
- environment and dependency versions
- ORCA description commit
- dex-retargeting commit
- Isaac Sim / Isaac Lab versions
- baseline methods:
  - direct joint-angle mapping
  - AnyTeleop-style VectorOptimizer
  - raw trajectory replay
  - baseline filtering / purification
- unified logging schema:
  - perception signals
  - retargeted qpos
  - q_desired
  - q_command
  - q_actual
  - timestamps and packet intervals
  - tracking validity

### Gate 1 — Offline Uncertainty-Aware Framework Smoke Test

目標：不用 Isaac 或相機也能驗證資料讀寫、品質分析、jitter injection 與
purification pipeline。

目前相關程式：

- `scripts/generate_synthetic_orca_demo.py`
- `scripts/generate_sample_isaac_client_jsonl.py`
- `scripts/analyze_orca_demo_quality.py`
- `scripts/inject_orca_jitter.py`
- `scripts/purify_orca_demo.py`
- `scripts/compare_raw_purified_demo.py`
- `scripts/replay_orca_demo_isaac.py`
- `research/orca_research/metrics.py`
- `research/orca_research/filters.py`
- `research/orca_research/noise.py`
- `research/orca_research/io.py`
- `research/orca_research/reporting.py`

Smoke test:

```bash
python scripts/generate_synthetic_orca_demo.py \
  --output data/examples/synthetic_orca_demo.npz

python scripts/analyze_orca_demo_quality.py \
  --input data/examples/synthetic_orca_demo.npz \
  --output-dir results/examples/quality

python scripts/inject_orca_jitter.py \
  --input data/examples/synthetic_orca_demo.npz \
  --output data/examples/synthetic_orca_demo_noisy.npz

python scripts/purify_orca_demo.py \
  --input data/examples/synthetic_orca_demo_noisy.npz \
  --output data/processed/synthetic_orca_demo_purified.npz \
  --method butterworth \
  --cutoff-hz 6.0 \
  --figure results/examples/purified_preview.png

python scripts/compare_raw_purified_demo.py \
  --raw data/examples/synthetic_orca_demo_noisy.npz \
  --purified data/processed/synthetic_orca_demo_purified.npz \
  --output-dir results/examples/raw_vs_purified
```

後續應擴充：

- tracking dropout injection
- landmark outlier injection
- latency / packet jitter injection
- depth invalid ratio
- confidence-weighted synthetic data
- task phase labels such as approach / grasp / hold / release

### Gate 2 — ORCA Isaac UDP Validation and Joint-Response Characterization

目標：在 Isaac Lab 中確認 ORCA Hand v2 可被 UDP client 穩定接收與控制，並建立
joint-level response baseline。

目前相關程式：

- `isaaclab/orca_hand_cfg.py`
- `isaaclab/orca_anyteleop_client.py`
- `teleop/send_orca_static_test.py`
- `teleop/send_orca_named_joint_sweep.py`
- `scripts/record_orca_udp_packets.py`

已完成部分：

- Isaac ORCA articulation import。
- 17-joint name mapping。
- UDP packet receive path。
- joint-limit clamp。
- command slew-rate limiting。
- `q_desired` / `q_command` / `q_actual` recording。

仍需完成：

- 每個 joint 的 named sweep validation。
- q_command to q_actual tracking error。
- step response。
- rise time / settling time。
- command-to-actual delay。
- velocity limit 對 response 的影響。
- open-hand neutral pose calibration。

### Gate 3 — AnyTeleop Baseline Reproduction

目標：明確復現 AnyTeleop 的 hand-only baseline，並將其移植到 ORCA Hand v2。

#### Gate 3A — AnyTeleop Hand-Only Reproduction in Isaac

Pipeline:

```text
D435i RGB camera
  -> MediaPipe / SingleHandDetector
  -> dex-retargeting VectorOptimizer
  -> ORCA joint names + qpos
  -> UDP
  -> Isaac Lab ORCA client
  -> ORCA Hand v2 Right in Isaac
```

目前相關程式：

- `teleop/orca_anyteleop_server.py`
- `config/retargeting/orca_v2_right_vector.yml`
- `config/retargeting/orca_v2_right_vector_virtual_tip.yml`
- `config/retargeting/orca_v2_vector_alignment_virtual_tip.yaml`
- `config/orca_v2_retarget_frames.yaml`
- `config/orca_v2_joint_semantics.yaml`
- `tools/test_continuous_hand_tracking.py`
- `tools/test_orca_dex_config.py`
- `tools/test_orca_dex_retarget_once.py`
- `tools/inspect_orca_robot_vectors.py`
- `tools/inspect_orca_fingertip_meshes.py`
- `tools/calibrate_orca_vector_alignment.py`

目前已達成第一筆 valid packet 到 Isaac，但 continuous control 尚未驗證。

#### Gate 3B — AnyTeleop Reproduction Debugging

需要優先解決：

- MediaPipe tracking valid / invalid 統計。
- UDP send / receive rate 統計。
- retargeted qpos consecutive logging。
- D435i RGB video node verification。
- `SeqRetargeting` initial qpos 與 ORCA open-hand pose mismatch。
- VectorOptimizer scale / frame alignment。
- retargeting filter / low-pass behavior。
- first-frame contraction problem。

#### Gate 3C — Physical ORCA Hand Safety Bridge

在連接實體 ORCA Hand 前必須完成：

- `orca_core` integration。
- semantic joint names to motor IDs mapping。
- joint limits。
- velocity limits。
- command timeout。
- tracking-loss watchdog。
- NaN / invalid-command rejection。
- safe neutral or open-hand behavior。
- emergency stop。
- motor current / temperature monitoring if available。
- tendon calibration / tension check procedure。

#### Gate 3D — AnyTeleop-Style Control of Physical ORCA Hand v2

目標：使用與 Isaac baseline 相同的 retargeting server，改由 physical ORCA client
接收 command。

需要記錄：

- q_desired from retargeting。
- q_command after safety layer。
- motor / joint feedback if available。
- packet timestamp。
- command-to-feedback latency。
- tracking validity。
- failure / watchdog events。

這一關完成後，才算真正完成「AnyTeleop 操作實體 ORCA Hand v2」的 baseline reproduction。

### Gate 4 — Formal Static, Dynamic, and Industrial Teaching Data Collection

目標：在 baseline 可穩定運作後，建立 formal dataset，而不是只保留臨時 debug log。

資料類型：

- Static gestures:
  - open hand
  - closed fist
  - thumb-index pinch
  - three-finger pinch
  - object holding pose
- Dynamic gestures:
  - open-to-fist
  - open-to-pinch
  - slow finger flexion
  - fast finger flexion
  - repeated grasp-release
- Industrial teaching tasks:
  - washer / nut pick-and-place
  - small connector pinch
  - object placement into fixture
  - button press
  - knob rotation if the hand is stable enough

每筆資料建議記錄：

- task name
- operator ID
- lighting condition
- background condition
- object type
- success / failure
- failure reason
- method name
- raw / filtered / proposed output source

### Gate 5 — Perception / Retargeting / Robot-Response Error Decomposition

目標：將 AnyTeleop baseline 在 ORCA Hand v2 上的問題拆成可量化來源。

Perception-side:

- landmark jitter。
- tracking loss。
- handedness confidence。
- depth invalid ratio。
- 2D / 3D landmark discontinuity。

Retargeting-side:

- optimizer residual。
- bad initialization。
- human-to-robot scale mismatch。
- vector-frame alignment error。
- q_desired jump。
- per-finger vector contribution。

Robot / control-side:

- q_command to q_actual delay。
- tracking RMSE / MAE。
- velocity clamp events。
- saturation / joint-limit events。
- tendon-driven response mismatch。
- physical motor feedback delay if available。

這一關的核心問題是：

> ORCA teleoperation 不穩主要來自 perception、retargeting，還是 robot response？

### Gate 6 — Robust Retargeting and Demonstration Purification Benchmark

目標：在 AnyTeleop-style baseline 之上加入 proposed method，並與 baseline filters 比較。

Baseline methods:

- raw AnyTeleop-style VectorOptimizer output。
- moving average。
- median。
- Savitzky-Golay。
- Butterworth。
- outlier + Butterworth。
- velocity clamp。

Proposed method 可包含：

- landmark validity weighting。
- depth validity weighting。
- per-finger confidence。
- Huber loss for outlier vectors。
- adaptive temporal regularization。
- tracking-loss hold / decay-to-neutral。
- task-phase-aware smoothing。
- ORCA response-aware compensation after Gate 5 analysis。

評估指標：

- velocity RMS。
- acceleration RMS。
- jerk RMS。
- total variation。
- high-frequency energy ratio。
- spectral entropy。
- q_command to q_actual tracking RMSE。
- phase delay。
- task success rate。
- object slip / contact instability if measurable。

### Gate 7 — Task-Level Replay and Industrial Application Evaluation

目標：用任務層級結果收尾，而不是只證明 trajectory 比較平滑。

Simulation evaluation:

- raw vs filtered vs proposed replay。
- grasp success rate。
- contact stability。
- final object pose error。
- trajectory smoothness。
- q_command to q_actual tracking error。

Physical ORCA evaluation:

- small-object pinch。
- washer / nut pick-and-place。
- button press。
- fixture placement。

比較項目：

- baseline AnyTeleop-style retargeting。
- baseline + purification。
- proposed uncertainty-aware retargeting / purification。

最終論文應能回答：

> Proposed method 是否提升 ORCA Hand v2 在工業示教任務中的穩定性、可重播性與任務成功率？

---

## 4. Target System

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
- Docker
- Git

Additional integration targets:

- `dex-retargeting`
- Pinocchio
- SciPy optimization tools
- `orca_core` for physical ORCA Hand control

---

## 5. Teleoperation Pipeline

```text
Intel RealSense D435i
        |
        v
RGB / aligned depth images
        |
        v
MediaPipe hand detection
        |
        | 21 image / world landmarks
        | tracking validity
        v
Human hand representation
        |
        | human joint angles
        | task-space vectors
        v
Human-to-ORCA retargeting
        |
        | direct mapping baseline
        | AnyTeleop-style VectorOptimizer baseline
        | proposed robust retargeting
        v
Command safety and filtering
        |
        | joint limits
        | velocity limits
        | command timeout
        | tracking-loss handling
        | jitter suppression
        |
        +-----------------------------+
        |                             |
        v                             v
Isaac Lab ORCA Hand          Physical ORCA Hand v2
        |                             |
        v                             v
q_desired / q_command /      motor / joint feedback
q_actual logging             and safety events
```

---

## 6. Repository Structure

Implemented or currently present files:

```text
repo_inspect/
├── README.md
├── THIRD_PARTY_VERSIONS.md
├── requirements-research.txt
│
├── assets/
│   └── urdf_work/
│       └── retarget/
│           └── orcahand_right_retarget.urdf
│
├── config/
│   ├── orca_v2_joint_semantics.yaml
│   ├── orca_v2_retarget_frames.yaml
│   ├── research_orca_quality.yaml
│   └── retargeting/
│       ├── orca_v2_right_vector.yml
│       ├── orca_v2_right_vector_virtual_tip.yml
│       └── orca_v2_vector_alignment_virtual_tip.yaml
│
├── docker/
│   └── retarget/
│       └── Dockerfile
│
├── docs/
│   ├── ANYTELEOP_ORCA_PROGRESS_20260812.md
│   ├── ORCA_RESEARCH_FRAMEWORK.md
│   └── ORCA_V2_URDF_IMPORT_STATUS.md
│
├── isaaclab/
│   ├── orca_anyteleop_client.py
│   └── orca_hand_cfg.py
│
├── research/
│   └── orca_research/
│       ├── filters.py
│       ├── io.py
│       ├── metrics.py
│       ├── noise.py
│       └── reporting.py
│
├── scripts/
│   ├── analyze_orca_demo_quality.py
│   ├── compare_raw_purified_demo.py
│   ├── generate_sample_isaac_client_jsonl.py
│   ├── generate_synthetic_orca_demo.py
│   ├── inject_orca_jitter.py
│   ├── purify_orca_demo.py
│   ├── record_orca_udp_packets.py
│   └── replay_orca_demo_isaac.py
│
├── teleop/
│   ├── orca_anyteleop_server.py
│   ├── send_orca_named_joint_sweep.py
│   └── send_orca_static_test.py
│
└── tools/
    ├── build_orca_retarget_urdf.py
    ├── calibrate_orca_vector_alignment.py
    ├── inspect_orca_fingertip_meshes.py
    ├── inspect_orca_robot_vectors.py
    ├── inspect_orca_usd.py
    ├── sanitize_orca_urdf_assets.py
    ├── test_continuous_hand_tracking.py
    ├── test_orca_dex_config.py
    └── test_orca_dex_retarget_once.py
```

Generated files and large experimental outputs should remain untracked:

- `data/`
- `results/`
- `logs/`
- `outputs/`
- `assets/usd/`
- `assets/urdf_work/isaac_package/`
- external repositories under `external/`

---

## 7. ORCA Model and Joint Audit

The official ORCA Hand model is obtained from:

```text
https://github.com/orcahand/orcahand_description
```

The target model is:

```text
ORCA Hand v2 Right
```

The ORCA URDF describes links, kinematic joints, joint axes, joint limits,
visual meshes, mass and inertia. It does not fully define every control-side
semantic required for teleoperation:

- ORCA semantic joint names
- FeeTech motor IDs
- HL2915 / HL3930 assignments
- motor inversion
- hardware neutral positions
- tendon calibration
- physical joint range of motion
- Isaac joint ordering
- sim-to-real direction consistency

The project therefore needs alignment among:

```text
MediaPipe human-hand representation
        |
        v
ORCA URDF kinematic joints
        |
        v
Isaac Lab articulation joints
        |
        v
orca_core semantic joints and FeeTech motors
```

Current supporting files:

- `config/orca_v2_joint_semantics.yaml`
- `config/orca_v2_retarget_frames.yaml`
- `tools/inspect_orca_usd.py`
- `tools/build_orca_retarget_urdf.py`
- `tools/sanitize_orca_urdf_assets.py`

---

## 8. AnyTeleop-Style Retargeting Baseline

The current baseline uses `dex-retargeting` and its VectorOptimizer-style
task-space retargeting. Instead of directly mapping human joint angles to robot
joint angles, the optimizer compares task-space vectors between the human hand
and the robot hand.

Conceptually:

```text
human reference vectors
        |
        v
scale / frame alignment
        |
        v
VectorOptimizer
        |
        v
ORCA qpos
```

Example robot vectors:

- palm to thumb tip
- palm to index tip
- palm to middle tip
- palm to ring tip
- palm to pinky tip
- intermediate finger segment vectors
- thumb-index pinch vector

Current retargeting configs:

- `config/retargeting/orca_v2_right_vector.yml`
- `config/retargeting/orca_v2_right_vector_virtual_tip.yml`
- `config/retargeting/orca_v2_vector_alignment_virtual_tip.yaml`

Current known issue:

```text
First valid retargeting command
        |
        v
ORCA Hand quickly flexes / contracts
        |
        v
The hand then appears to stop following human motion
```

Likely causes:

- `SeqRetargeting` initial qpos is not calibrated to ORCA open-hand pose。
- vector scale mismatch。
- frame alignment mismatch。
- MediaPipe tracking validity after first frame。
- camera stream selection。
- command update frequency。
- retargeting low-pass behavior。

---

## 9. Recording and Analysis

The Isaac Lab client can record one row per simulation step:

```text
t_sim
timestamp
seq
packet_timestamp
tracking_valid
joint_names
q_desired
q_command
q_actual
```

Definitions:

- `q_desired`: latest retargeting command after joint-limit clamping。
- `q_command`: command sent to Isaac after slew-rate limiting。
- `q_actual`: PhysX articulation joint position。

The retargeting server can record:

- MediaPipe validity。
- raw human reference vectors。
- aligned ORCA-frame reference vectors。
- retargeted ORCA qpos。

Implemented quality metrics:

- position RMS。
- velocity RMS。
- acceleration RMS。
- jerk RMS。
- total variation。
- high-frequency energy ratio。
- spectral entropy。
- tracking RMSE / MAE / max absolute error when `q_actual` is available。

Implemented purification methods:

- `none`
- `moving_average`
- `median`
- `savgol`
- `butterworth`
- `outlier_then_butterworth`
- `velocity_clamp`

See:

- `docs/ORCA_RESEARCH_FRAMEWORK.md`
- `config/research_orca_quality.yaml`

---

## 10. Common Commands

Run the Isaac Lab ORCA teleoperation client:

```bash
cd /home/lab606/Projects/orca_isaaclab_import

/home/lab606/IsaacLab/isaaclab.sh -p isaaclab/orca_anyteleop_client.py \
  --udp-host 0.0.0.0 \
  --udp-port 5006 \
  --record-path data/raw/isaac_client_session.jsonl
```

Run the retargeting server inside the dex-retargeting Docker environment:

```bash
python teleop/orca_anyteleop_server.py \
  --config config/retargeting/orca_v2_right_vector.yml \
  --alignment-config config/retargeting/orca_v2_vector_alignment_virtual_tip.yaml \
  --host 127.0.0.1 \
  --port 5006 \
  --camera /dev/video4 \
  --hand-type Right \
  --record-path data/raw/teleop_server_session.jsonl
```

Send a static test command:

```bash
python teleop/send_orca_static_test.py \
  --host 127.0.0.1 \
  --port 5006
```

Send named joint sweep commands:

```bash
python teleop/send_orca_named_joint_sweep.py \
  --host 127.0.0.1 \
  --port 5006
```

Analyze a recorded Isaac session:

```bash
python scripts/analyze_orca_demo_quality.py \
  --input data/raw/isaac_client_session.jsonl \
  --trajectory-key q_command \
  --output-dir results/isaac_client_quality
```

Purify a recorded trajectory:

```bash
python scripts/purify_orca_demo.py \
  --input data/raw/isaac_client_session.jsonl \
  --output data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --method outlier_then_butterworth \
  --cutoff-hz 6.0 \
  --figure results/isaac_client_purified_preview.png
```

Compare raw and purified trajectories:

```bash
python scripts/compare_raw_purified_demo.py \
  --raw data/raw/isaac_client_session.jsonl \
  --purified data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --output-dir results/isaac_client_raw_vs_purified
```

Replay a saved trajectory into Isaac through UDP:

```bash
python scripts/replay_orca_demo_isaac.py \
  --input data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --host 127.0.0.1 \
  --port 5006
```

---

## 11. Safety Notice

The current development stage focuses on simulation and baseline reproduction.

Before sending MediaPipe or optimizer outputs to the physical ORCA Hand, the
following must be implemented and verified:

- joint-limit enforcement。
- velocity limits。
- command timeout。
- tracking-loss watchdog。
- NaN and invalid-command rejection。
- motor-current monitoring if available。
- motor-temperature monitoring if available。
- emergency stop。
- safe neutral or open-hand behavior。
- tendon calibration and tension verification。

MediaPipe or optimizer output must not be sent directly to the physical ORCA Hand
without a safety layer.

---

## 12. Development Environment

Current known environment:

- Operating system: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 3060
- NVIDIA driver: 580.173.02
- CUDA: 13.0
- Isaac Sim: 5.1.x
- Isaac Lab version: 2.3.2
- Isaac Lab branch: main
- Isaac Lab commit: training-checkpoints-develop-16-gb4c3210247
- URDF importer: 2.4.31
- Host Python: 3.10.12
- Isaac runtime Python: 3.11
- ORCA description commit: b9b349a21ee0238c62b6cf92ae7597027867adf8

Keep this section consistent with `THIRD_PARTY_VERSIONS.md`.

---

## 13. References

Conceptual references:

- DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System
- AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System

Official ORCA resources:

- ORCA Hand description: `https://github.com/orcahand/orcahand_description`
- ORCA Hand core: `https://github.com/orcahand/orca_core`

Dependency versions are recorded in:

```text
THIRD_PARTY_VERSIONS.md
```

---

## 14. Git Policy

Commit:

- source code
- configuration files
- audit tools
- modified URDF working copies
- Isaac Lab asset configuration
- documentation
- analysis scripts
- small reproducible examples

Do not commit:

- external repositories
- generated datasets
- generated figures
- videos
- Isaac / Omniverse caches
- ROS bags
- large logs
- virtual environments


