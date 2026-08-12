#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect a converted ORCA Hand USD file.

This script launches Isaac Sim through Isaac Lab's AppLauncher before
importing pxr modules. It performs read-only structural inspection and
does not start physics simulation or command any robot joints.

Reported information:
- Default prim
- Articulation roots
- Rigid bodies
- Collision prims
- Visual mesh prims
- Revolute, prismatic and fixed joints
- Joint body relationships
- Revolute-joint axes and limits
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# ---------------------------------------------------------------------
# Parse command-line arguments before launching Isaac Sim.
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Inspect the converted ORCA Hand USD."
)

parser.add_argument(
    "--usd",
    required=True,
    type=Path,
    help="Path to the converted ORCA Hand USD file.",
)

# Add Isaac Lab launcher arguments such as --headless.
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------------------------------------------------
# Launch Isaac Sim / Omniverse Kit.
# pxr and omni imports must occur after this point.
# ---------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics


def relation_targets(relation: Usd.Relationship) -> str:
    """Convert USD relationship targets into printable text."""
    targets = relation.GetTargets()

    if not targets:
        return "-"

    return ", ".join(str(target) for target in targets)


def optional_float(value) -> float | None:
    """Safely convert a USD attribute value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def degree_to_radian(value: float | None) -> float | None:
    """Convert degrees to radians while preserving missing values."""
    if value is None:
        return None

    return math.radians(value)


def main() -> None:
    usd_path = args_cli.usd.expanduser().resolve()

    if not usd_path.is_file():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))

    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    default_prim = stage.GetDefaultPrim()

    articulation_roots: list[str] = []
    rigid_bodies: list[str] = []
    colliders: list[str] = []
    meshes: list[str] = []

    revolute_joints: list[Usd.Prim] = []
    prismatic_joints: list[Usd.Prim] = []
    fixed_joints: list[Usd.Prim] = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(prim.GetPath()))

        if prim.IsA(UsdGeom.Mesh):
            meshes.append(str(prim.GetPath()))

        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim)

        elif prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic_joints.append(prim)

        elif prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(prim)

    print("=" * 88)
    print("ORCA Hand USD Inspection")
    print("=" * 88)

    print(f"USD path                  : {usd_path}")
    print(
        "Default prim              : "
        f"{default_prim.GetPath() if default_prim else None}"
    )
    print(f"Articulation roots        : {len(articulation_roots)}")
    print(f"Rigid bodies              : {len(rigid_bodies)}")
    print(f"Collision prims           : {len(colliders)}")
    print(f"Visual mesh prims         : {len(meshes)}")
    print(f"Revolute joints           : {len(revolute_joints)}")
    print(f"Prismatic joints          : {len(prismatic_joints)}")
    print(f"Fixed joints              : {len(fixed_joints)}")

    print()
    print("Articulation roots")
    print("-" * 88)

    if articulation_roots:
        for path in articulation_roots:
            print(f"  {path}")
    else:
        print("  No articulation root API found.")

    print()
    print("Revolute joints")
    print("-" * 88)

    if not revolute_joints:
        print("  No revolute joints found.")

    for index, prim in enumerate(revolute_joints):
        joint = UsdPhysics.RevoluteJoint(prim)

        axis = joint.GetAxisAttr().Get()

        # USD Physics stores revolute-joint limits in degrees.
        lower_deg = optional_float(joint.GetLowerLimitAttr().Get())
        upper_deg = optional_float(joint.GetUpperLimitAttr().Get())

        lower_rad = degree_to_radian(lower_deg)
        upper_rad = degree_to_radian(upper_deg)

        body0 = relation_targets(joint.GetBody0Rel())
        body1 = relation_targets(joint.GetBody1Rel())

        print(f"[{index:02d}] {prim.GetName()}")
        print(f"     path       : {prim.GetPath()}")
        print(f"     body0      : {body0}")
        print(f"     body1      : {body1}")
        print(f"     axis       : {axis}")
        print(f"     lower deg  : {lower_deg}")
        print(f"     upper deg  : {upper_deg}")
        print(f"     lower rad  : {lower_rad}")
        print(f"     upper rad  : {upper_rad}")

    print()
    print("Collision prims")
    print("-" * 88)

    if colliders:
        for path in colliders:
            print(f"  {path}")
    else:
        print("  No collision prims found.")

    print()
    print("Rigid bodies")
    print("-" * 88)

    if rigid_bodies:
        for path in rigid_bodies:
            print(f"  {path}")
    else:
        print("  No rigid-body APIs found.")

    print()
    print("Inspection completed successfully.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close Isaac Sim cleanly, including when inspection fails.
        simulation_app.close()
