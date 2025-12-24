# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch 

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import curriculums as mdp_curriculums
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp import rewards as mdp_rewards


##
# Scene definition
##


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Scene with fixed-base G1 upper body and two cubes on a table."""

    # Robot (fixed base)
    robot: ArticulationCfg = G1_29DOF_CFG

    # Left cube
    left_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftCube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.20, 0.45, 0.70],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube_1")],
        ),
    )

    # Right cube
    right_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightCube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.20, 0.45, 0.70],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube_2")],
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        # Fix the robot base
        self.robot.spawn.articulation_props.fix_root_link = True


##
# Actions
##


@configclass
class ActionsCfg:
    """Joint-position actions for G1 upper-body."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_.*",
            "left_elbow_.*",
            "left_wrist_.*",
            "right_shoulder_.*",
            "right_elbow_.*",
            "right_wrist_.*",
        ],
        scale=0.5,  # [-0.5, 0.5] around default pose
        use_default_offset=True,
    )


##
# Observations
##


@configclass
class ObservationsCfg:
    """Flat 1-D observation vector for RSL-RL."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations used by both actor and critic."""

        # robot
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # cube positions in world frame
        left_object_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("left_object")},
        )
        right_object_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("right_object")},
        )

        def __post_init__(self):
            # IMPORTANT: RSL-RL expects a flat (N, D) tensor
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Terminations
##


@configclass
class TerminationsCfg:
    """Simple terminations."""

    # episode timeout handled by env
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # optional: terminate if a cube falls below the table
    left_object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("left_object")},
    )
    right_object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("right_object")},
    )


@configclass
class EventCfg:
    """Reset events for the environment."""

    # Reset whole scene to default initial state (robot + cubes + table)
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
    )


##
# Rewards
##

##
# Custom reward helpers (grasp/hold)
##


def _hand_object_gaussian_distance_reward(
    env,
    hand_link_name: str,
    object_cfg: SceneEntityCfg,
    std: float = 0.05,
    lift_height: float | None = None,
) -> torch.Tensor:
    """Positive reward when the given hand is close to the object (and optionally lifted)."""
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]

    # Positions in env frame
    body_names = robot.data.body_names
    hand_idx = body_names.index(hand_link_name)
    hand_pos = robot.data.body_pos_w[:, hand_idx] - env.scene.env_origins
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    # Distance
    dist = torch.linalg.norm(hand_pos - obj_pos, dim=1)

    # Gaussian shaping: 1 when on top, ~0 when far
    rew = torch.exp(-0.5 * (dist / std) ** 2)

    # Optional: only count when object is lifted
    if lift_height is not None:
        lifted = (obj_pos[:, 2] > lift_height).float()
        rew = rew * lifted

    return rew


def left_hand_grasp_hold(env, std: float = 0.05, lift_height: float = 0.75) -> torch.Tensor:
    return _hand_object_gaussian_distance_reward(
        env,
        hand_link_name="left_wrist_yaw_link",
        object_cfg=SceneEntityCfg("left_object"),
        std=std,
        lift_height=lift_height,
    )


def right_hand_grasp_hold(env, std: float = 0.05, lift_height: float = 0.75) -> torch.Tensor:
    return _hand_object_gaussian_distance_reward(
        env,
        hand_link_name="right_wrist_yaw_link",
        object_cfg=SceneEntityCfg("right_object"),
        std=std,
        lift_height=lift_height,
    )



@configclass
class RewardsCfg:
    """Shaping rewards for two-cube task."""

    # alive bonus
    alive = RewTerm(func=mdp_rewards.is_alive, weight=1.0)

    # penalty for termination (except timeout)
    terminating = RewTerm(func=mdp_rewards.is_terminated, weight=-5.0)

    # smooth motions
    joint_vel = RewTerm(
        func=mdp_rewards.joint_vel_l2,
        weight=-5e-2,  # starts mild, will be increased by curriculum
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # small action penalty
    action_l2 = RewTerm(
        func=mdp_rewards.action_l2,
        weight=-1e-3,  # starts very small, curriculum will increase
    )

    # encourage higher cube heights (simple lifting)
    left_cube_height = RewTerm(
        func=mdp_rewards.base_height_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("left_object"),
            "target_height": 0.80,
        },
    )

    right_cube_height = RewTerm(
        func=mdp_rewards.base_height_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("right_object"),
            "target_height": 0.80,
        },
    )

    # NEW: grasp + hold rewards
    left_grasp_hold = RewTerm(
        func=left_hand_grasp_hold,
        weight=8.0,
        params={"std": 0.05, "lift_height": 0.75},
    )

    right_grasp_hold = RewTerm(
        func=right_hand_grasp_hold,
        weight=8.0,
        params={"std": 0.05, "lift_height": 0.75},
    )


##
# Curriculum (for velocities / actions)
##


@configclass
class CurriculumCfg:
    """Curriculum terms: gradually increase penalties to discourage extreme motions."""

    joint_vel = CurrTerm(
        func=mdp_curriculums.modify_reward_weight,
        params={
            "term_name": "joint_vel",
            "weight": -1e-1,   # target weight (stronger penalty)
            "num_steps": 20000,
        },
    )

    action_l2 = CurrTerm(
        func=mdp_curriculums.modify_reward_weight,
        params={
            "term_name": "action_l2",
            "weight": -5e-2,   # target weight (discourage large actions)
            "num_steps": 20000,
        },
    )


##
# Env cfg
##


@configclass
class MultiObjectEnvCfg(ManagerBasedRLEnvCfg):
    """G1 fixed-base, two-cube environment for RSL-RL."""

    # Scene
    scene: MultiObjectSceneCfg = MultiObjectSceneCfg(
        num_envs=512,  # you can override from Hydra/CLI
        env_spacing=2.5,
        replicate_physics=True,
    )

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # No commands/events/curriculum for now
    commands = None
    events  : EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        # General
        self.decimation = 2
        self.episode_length_s = 5.0

        # Viewer
        self.viewer.eye = (6.0, 0.0, 4.0)

        # Sim
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
