# Gate 3A ORCA AnyTeleop Server Refactor Guide

## Goal

Refactor the original ~1,877-line `teleop/orca_anyteleop_server.py` into small modules **without changing the teleoperation algorithm or Gate 3A JSONL schema**.

The refactored pipeline remains:

```text
OpenCV / D435i RGB
  -> SingleHandDetector / MediaPipe
  -> 21x3 wrist-centered landmarks
  -> raw human vectors
  -> Human -> ORCA rotation + scale
  -> dex-retargeting VectorOptimizer
  -> 17-DoF qpos
  -> JSONL
  -> optional UDP -> Isaac Lab
```

`--dry-run` still runs every upstream stage and only disables UDP transmission.

## Files

All six Python files belong in the repository `teleop/` directory.

### `orca_anyteleop_server.py`
Thin orchestration entry point. It contains the per-frame pipeline and makes the execution order easy to audit.

### `orca_server_setup.py`
One-time initialization: CLI, pinned `SingleHandDetector`, alignment selection, dex-retargeting construction, ORCA initial pose, fixed-wrist validation, and OpenCV camera setup.

### `orca_vector_alignment.py`
Loads/validates the alignment YAML and applies:

```text
V_orca = scale * V_human @ R.T
```

No filtering or optimization is performed here.

### `orca_gate3a_recorder.py`
Owns Gate 3A JSONL recording and converts MediaPipe `NormalizedLandmarkList` into numeric `(21,3)` arrays. Invalid tracking frames remain in the timeline.

### `orca_runtime_diagnostics.py`
Owns all startup/runtime console diagnostics, including `[SERVER]`, `[HVEC]`, `[ALIGN]`, and `[QPOS]`. It is observation-only.

### `orca_udp.py`
Owns valid/invalid UDP packet schemas and transmission. No other module should call `sock.sendto()` directly.

## Behavior intentionally preserved

- Same CLI names and defaults.
- Same virtual-tip/retargeting YAML supplied by `--config`.
- Same alignment YAML supplied by `--alignment-config`.
- Same 17-DoF validation and fixed wrist.
- Same zero/clamped initial qpos.
- Same human vector construction from `optimizer.target_link_human_indices`.
- Same Human -> ORCA alignment equation.
- Same `retargeting.retarget(..., fixed_qpos=fixed_qpos)` call.
- Same valid UDP packet fields.
- Same invalid UDP packet fields.
- Same Gate 3A valid/invalid JSONL schema.
- Same dry-run policy: recording and optimizer continue, UDP stops.
- Same optional OpenCV `--show` path.
- Same `retargeting.verbose()` shutdown output.

## Safe replacement procedure

From the host repository:

```bash
cd ~/Projects/orca_isaaclab_import

cp teleop/orca_anyteleop_server.py \
   teleop/orca_anyteleop_server_pre_refactor.py
```

Copy the six refactored Python files into `teleop/`.

Then run inside the Gate 3 Docker container:

```bash
cd /workspace/project

python -m py_compile \
  teleop/orca_anyteleop_server.py \
  teleop/orca_server_setup.py \
  teleop/orca_vector_alignment.py \
  teleop/orca_gate3a_recorder.py \
  teleop/orca_runtime_diagnostics.py \
  teleop/orca_udp.py
```

No output means syntax PASS.

Check CLI compatibility:

```bash
python teleop/orca_anyteleop_server.py --help
```

The existing flags must still include:

```text
--config
--alignment-config
--camera
--host
--port
--width
--height
--fps
--hand-type
--selfie
--show
--wrist-position
--record-path
--dry-run
--duration
```

## Regression smoke test

Do not overwrite the successful `smoke_r1.jsonl`. Run a new `smoke_r2.jsonl`:

```bash
python teleop/orca_anyteleop_server.py \
  --config \
  config/retargeting/orca_v2_right_vector_virtual_tip.yml \
  --alignment-config \
  config/retargeting/orca_v2_vector_alignment_virtual_tip.yaml \
  --camera /dev/video4 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --hand-type Right \
  --dry-run \
  --duration 10 \
  --record-path \
  data/gate3a/raw/smoke_r2.jsonl
```

Do not use `--show` until the container Qt/XCB issue is independently resolved.

### Required regression checks

1. The program reaches `Requested experiment duration reached` without traceback.
2. `Dry run : True` and `UDP enabled : False`.
3. `read_fail=0` during normal capture.
4. Valid tracking is high for a clean static open-hand smoke test.
5. `retarget_fail=0`.
6. `sent=0` in dry-run.
7. JSONL sequence numbers are monotonic and contiguous for captured frames.
8. Every valid row contains finite arrays with these shapes:
   - `joint_pos`: `(21,3)`
   - `keypoint_2d`: `(21,3)`
   - `reference_vectors_raw`: `(10,3)`
   - `reference_vectors_aligned`: `(10,3)`
   - `positions_rad`: `(17,)`
9. Raw/aligned vector length ratio remains approximately the alignment scale.
10. Human vector semantic order remains:
    - HV02 = middle tip
    - HV03 = ring tip
    - HV07 = middle PIP
    - HV08 = ring PIP

## Git recommendation

Only after `smoke_r2` passes:

```bash
git add teleop/orca_anyteleop_server.py \
        teleop/orca_server_setup.py \
        teleop/orca_vector_alignment.py \
        teleop/orca_gate3a_recorder.py \
        teleop/orca_runtime_diagnostics.py \
        teleop/orca_udp.py

git commit -m "Refactor ORCA AnyTeleop server without changing Gate 3A behavior"
git push origin main
```

Keep raw `smoke_r1.jsonl` and `smoke_r2.jsonl` out of Git unless your repository policy explicitly tracks small experiment logs.
