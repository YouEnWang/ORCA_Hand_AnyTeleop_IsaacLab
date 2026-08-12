## ORCA Hand Description

- Repository: https://github.com/orcahand/orcahand_description
- Branch: main
- Commit: `b9b349a21ee0238c62b6cf92ae7597027867adf8`
- Model: ORCA Hand v2 Right
- Original URDF:
  `v2/models/urdf/orcahand_right.urdf`
- Import verification date: 2026-08-06
- Import status: Visually displayed in Isaac Sim 5.1.0
- Compatibility note:
  Referenced mesh filenames require sanitization before import into
  the current Isaac Sim environment.
- Sanitization tool:
  `tools/sanitize_orca_urdf_assets.py`

## Isaac Lab

- Repository: https://github.com/isaac-sim/IsaacLab
- Branch: main
- Commit: `b4c3210247`
- Isaac Lab version: 2.3.2
- Isaac Sim runtime: 5.1.0

## dex-retargeting

- Repository: https://github.com/dexsuite/dex-retargeting
- Commit: 3f56141bc8bd2760d5e452e382937269554ebb21
- Branch: main
- Origin: AnyTeleop
- Purpose: Human-to-robot dexterous hand retargeting

## sim-web-visualizer

- Repository: https://github.com/NVlabs/sim-web-visualizer
- Commit: 20b02f3380b7872bf262ac26859879d1c1385671
- Branch: main
- Origin: AnyTeleop
- Purpose: Web-based visualizer for simulation environments

## AnyTeleop Retargeting Container

- Base image: `python:3.10-slim`
- Docker image: `orca-anyteleop-retarget:c9de8bddf3f0`
- PyTorch: CPU-only
- MediaPipe: `0.10.21`
- dex-retargeting:
  `3f56141bc8bd2760d5e452e382937269554ebb21`

The retargeting environment is intentionally isolated from the
Isaac Lab Python environment.