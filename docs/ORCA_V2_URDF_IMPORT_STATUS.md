# ORCA Hand v2 Right URDF Import Status

## 1. Overview

This document records the current status of importing the official
ORCA Hand v2 Right URDF into Isaac Sim through the Isaac Lab URDF
conversion tool.

Import date:

- 2026-08-06

Current result:

- ORCA Hand v2 Right is visually displayed in Isaac Sim.
- The forearm, palm, thumb and four fingers are visible.
- The model hierarchy is present in the Isaac Sim Stage.
- The converted USD file was generated successfully.
- The previous fatal `RuntimeError: Used null prim` no longer occurs.

Current stage:

> Visual and structural import succeeded. Formal USD articulation,
> joint-count, joint-limit, collision and joint-motion verification
> have not yet been completed.

---

## 2. Development Environment

- Operating system: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 3060
- NVIDIA driver: 580.173.02
- Driver-supported CUDA: 13.0
- Isaac Sim: 5.1.0
- Isaac Lab version: 2.3.2
- Isaac Lab branch: main
- Isaac Lab commit: `b4c3210247`
- Host Python: 3.10.12
- Isaac runtime Python: 3.11
- ORCA description commit:
  `b9b349a21ee0238c62b6cf92ae7597027867adf8`

---

## 3. Source Asset

Official repository:

```text
https://github.com/orcahand/orcahand_description
```

Original ORCA Hand model:

```text
external/orcahand_description/
└── v2/models/urdf/orcahand_right.urdf
```

The original official repository is preserved without modification.

An Isaac-compatible working copy was created under:

```text
assets/urdf_work/isaac_package/
└── orcahand_description/
    └── v2/models/urdf/orcahand_right.urdf
```

## 4. Generated USD

The converted USD was generated at:

```text
assets/usd/orca_v2_right_sanitized/orcahand_right.usd
```

The current USD was generated for inspection and testing.

The generated USD has not yet been formally validated for:

- Expected 17 controllable DoFs
- Articulation root configuration
- Joint ordering
- Joint limits
- Joint direction
- Position-drive behavior
- Collision geometry
- Self-collision
- Physical accuracy

## 5. URDF Conversion Command

The following command was used:

```bash
cd ~/IsaacLab

./isaaclab.sh -p "$URDF_CONVERTER" \
  "$ORCA_URDF_ISAAC" \
  "$ORCA_PROJECT/assets/usd/orca_v2_right_sanitized/orcahand_right.usd" \
  --fix-base \
  --joint-stiffness 0.0 \
  --joint-damping 0.0 \
  --joint-target-type none
```

The main importer configuration was:

```text
fix_base: True
merge_fixed_joints: False
make_instanceable: True
joint target type: none
joint stiffness: 0.0
joint damping: 0.0
self_collision: False
collision_from_visuals: False
collider_type: convex_hull
```

## 6. Required Asset Sanitization

### 6.1 Original problem

The original ORCA Hand CAD-exported assets contain hyphens in mesh
filenames and names, for example:

```text
ForeArmStructure-Model
TopTower-Model
R-Carpals
P-FingerTipAssembly
```

During the first import attempt, Isaac Sim attempted to use a mesh
filename containing a hyphen as a USD prim identifier. This generated:

```text
Ill-formed SdfPath
RuntimeError: Used null prim
```

As a result, the first USD conversion was invalid.

### 6.2 Applied workaround

A working copy of the official ORCA description package was created.

The script:

```text
tools/sanitize_orca_urdf_assets.py
```

was used to:

1. Inspect mesh references in the copied URDF.
2. Rename referenced mesh filenames containing unsupported characters.
3. Replace hyphens and other unsupported characters with underscores.
4. Update the corresponding URDF mesh references.
5. Preserve the official external repository without modification.

Example:

```text
ForeArmStructure-Model.stl
```

was changed to:

```text
ForeArmStructure_Model.stl
```

## 7. Current Warnings

### 7.1 Automatic USD Path Renaming

Example:

```text
The path R-Carpals_8d1f1041 is not a valid usd path,
modifying to R_Carpals_8d1f1041
```

Similar warnings appear for multiple ORCA links and joints.

**Meaning**

Some link and joint names in the original URDF still contain hyphens.
The Isaac Sim importer automatically replaces these characters with
underscores when creating USD prim paths.

Example:

```text
URDF name:
R-Carpals_8d1f1041
```

becomes:

```text
USD name:
R_Carpals_8d1f1041
```

**Current assessment**

- Non-fatal
- USD conversion completed
- Visual model is present
- No immediate modification is required

**Required follow-up**

Both names must be recorded in the future joint bridge:

```text
Original URDF joint name
        ↕
Sanitized Isaac/USD joint name
        ↕
ORCA semantic joint name
```

This warning cannot be ignored when implementing joint-name mapping.

---

### 7.2 Non-Primary Joint Axis Warning

Warning:

```text
R_T_AP_a9723101_to_T_TP_R_1c2b802d:
Joint Axis is not body aligned with X, Y or Z primary axis.
Adjusting PhysX joint alignment to Axis X and reorienting bodies.
```

**Meaning**

The original URDF joint axis is not aligned directly with the local
X, Y or Z primary axis.

The importer therefore:

1. Aligns the PhysX revolute joint with a supported local primary axis.
2. Reorients the connected body frames.
3. Attempts to preserve the original joint-motion direction.

**Current assessment**

- Non-fatal
- Import completed
- This joint must receive additional validation

**Required follow-up**

During the joint sweep test, verify:

- Which anatomical ORCA joint this represents
- Positive and negative rotation direction
- Whether motion follows the intended mechanism
- Whether unexpected translation occurs
- Whether the original URDF joint limits are preserved
- Whether the thumb geometry remains correct

This warning should remain documented until the joint has been
validated experimentally.
