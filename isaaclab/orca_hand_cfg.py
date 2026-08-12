# -*- coding: utf-8 -*-

"""
Isaac Lab configuration for ORCA Hand v2 Right.

This configuration is for the first AnyTeleop integration test.

Important:
- The hand is fixed-base.
- No collision geometry is currently available.
- The actuator parameters are preliminary simulation-control values.
- They are NOT physical models of the FeeTech HL2915 / HL3930 motors.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ORCA_USD_PATH = (
    "/home/lab606/Projects/orca_isaaclab_import/"
    "assets/usd/orca_v2_right_sanitized/"
    "orcahand_right.usd"
)


ORCA_HAND_CFG = ArticulationCfg(

    prim_path="/World/ORCAHand",

    spawn=sim_utils.UsdFileCfg(

        usd_path=ORCA_USD_PATH,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
    ),

    # From our USD inspection:
    #
    # /orcahand_right/root_joint
    #
    # After spawning the USD under /World/ORCAHand,
    # the articulation root is relative to asset root.
    articulation_root_prim_path="/root_joint",

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(0.0, 0.0, 0.4),

        rot=(1.0, 0.0, 0.0, 0.0),

        joint_pos={
            ".*": 0.0,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),

    actuators={

        "orca_joints": ImplicitActuatorCfg(

            joint_names_expr=[
                ".*"
            ],

            # Initial simulation values only.
            effort_limit_sim=5.0,

            velocity_limit_sim=3.0,

            stiffness=30.0,

            damping=2.0,
        ),
    },

    soft_joint_pos_limit_factor=0.95,
)
