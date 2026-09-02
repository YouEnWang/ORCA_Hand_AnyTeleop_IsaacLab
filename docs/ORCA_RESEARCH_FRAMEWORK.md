# ORCA Teleoperation Demonstration Quality Framework

This document describes the first focused research framework for ORCA Hand v2
Right teleoperation data quality analysis and purification.

The first version intentionally does not include MuJoCo comparison or
across-hand transfer. It focuses on:

```text
D435i / MediaPipe
  -> AnyTeleop-style ORCA retargeting
  -> IsaacLab ORCA v2 right
  -> signal recording
  -> offline quality analysis
  -> demonstration purification
  -> raw vs purified comparison
  -> trajectory replay
```

## 1. Install Research Dependencies

Use your existing Linux IsaacLab and retargeting environments for simulation and
camera work. For offline analysis, install the lightweight research dependencies:

```bash
cd /home/lab606/Projects/orca_isaaclab_import
python3 -m venv .venv-research
source .venv-research/bin/activate
pip install -r requirements-research.txt
```

## 2. Smoke Test Without Isaac

This checks that the offline research framework works before using Isaac Sim:

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

Expected outputs:

```text
results/examples/quality/joint_quality_metrics.csv
results/examples/quality/quality_summary.json
results/examples/quality/trajectory_preview.png
results/examples/raw_vs_purified/raw_vs_purified_joint_metrics.csv
results/examples/raw_vs_purified/raw_vs_purified_summary.json
results/examples/raw_vs_purified/raw_vs_purified_preview.png
```

## 3. Record Isaac Client Signals

Run the IsaacLab client with recording enabled:

```bash
cd /home/lab606/Projects/orca_isaaclab_import

/home/lab606/IsaacLab/isaaclab.sh -p isaaclab/orca_anyteleop_client.py \
  --udp-host 0.0.0.0 \
  --udp-port 5006 \
  --record-path data/raw/isaac_client_session.jsonl
```

The client records one row per simulation step with:

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

```text
q_desired : latest retargeting command after joint-limit clamping
q_command : command sent to Isaac after slew-rate limiting
q_actual  : PhysX articulation joint position
```

## 4. Record Retargeting Server Signals

Run the retargeting server inside the dex-retargeting Docker container with:

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

The server records MediaPipe validity, raw human reference vectors, aligned
ORCA-frame reference vectors, and retargeted ORCA qpos.

## 5. Analyze A Recorded Isaac Session

```bash
python scripts/analyze_orca_demo_quality.py \
  --input data/raw/isaac_client_session.jsonl \
  --trajectory-key q_command \
  --output-dir results/isaac_client_quality
```

This produces joint-wise metrics including:

```text
velocity_rms
acceleration_rms
jerk_rms
total_variation
high_frequency_energy_ratio
spectral_entropy
tracking_rmse
tracking_mae
tracking_max_abs
```

## 6. Purify A Recorded Trajectory

```bash
python scripts/purify_orca_demo.py \
  --input data/raw/isaac_client_session.jsonl \
  --output data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --method outlier_then_butterworth \
  --cutoff-hz 6.0 \
  --figure results/isaac_client_purified_preview.png
```

Available first-version methods:

```text
none
moving_average
median
savgol
butterworth
outlier_then_butterworth
velocity_clamp
```

## 7. Compare Raw And Purified Data

```bash
python scripts/compare_raw_purified_demo.py \
  --raw data/raw/isaac_client_session.jsonl \
  --purified data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --output-dir results/isaac_client_raw_vs_purified
```

## 8. Replay A Saved Trajectory Into Isaac

Start the IsaacLab client first, then replay a trajectory through UDP:

```bash
python scripts/replay_orca_demo_isaac.py \
  --input data/processed/isaac_client_session_purified.npz \
  --trajectory-key q_command \
  --host 127.0.0.1 \
  --port 5006
```

This uses the same UDP packet schema as the AnyTeleop retargeting server:

```text
seq
timestamp
tracking_valid
joint_names
positions_rad
source
```

## 9. First Research Milestones

1. Record static gestures for 20-30 seconds.
2. Analyze simulator/client noise floor using `q_command` and `q_actual`.
3. Record dynamic gestures such as open-to-fist and thumb-index pinch.
4. Run controlled jitter injection on clean synthetic or slow trajectories.
5. Compare purification methods with signal-level metrics.
6. Replay raw and purified trajectories into Isaac.
7. Add task/contact phases once grasping and collision behavior are validated.

## 10. Git Policy

Commit source code, configs, small docs, and small example metadata.

Do not commit generated datasets, generated figures, videos, or Isaac caches.
The `.gitignore` file excludes `data/`, `results/`, `outputs/`, and `logs/`.

