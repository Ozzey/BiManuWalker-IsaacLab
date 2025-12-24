# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_multi_object_pick_place(
    env: ManagerBasedRLEnv,
    # separate configs for each cube
    left_object_cfg: SceneEntityCfg = SceneEntityCfg("left_object"),
    right_object_cfg: SceneEntityCfg = SceneEntityCfg("right_object"),
    # target region for each cube in *env-local* coordinates
    left_min_x: float = 0.40,
    left_max_x: float = 0.85,
    left_min_y: float = 0.35,
    left_max_y: float = 0.60,
    right_min_x: float = 0.40,
    right_max_x: float = 0.85,
    right_min_y: float = 0.35,
    right_max_y: float = 0.60,
    max_height: float = 1.10,
    max_vel: float = 0.20,
) -> torch.Tensor:
    """Success if BOTH cubes are placed in their target regions and are static.

    Conditions (per env):
      - each cube's (x, y) lies within its respective [min, max] bounds
      - each cube's height is below max_height
      - each cube's linear velocity is below max_vel in all axes

    All checks are done in coordinates relative to env.scene.env_origins.
    """

    # --- Left cube ---
    left_object: RigidObject = env.scene[left_object_cfg.name]
    left_pos = left_object.data.root_pos_w - env.scene.env_origins  # [N, 3]
    left_vel = torch.abs(left_object.data.root_vel_w)               # [N, 3]

    left_x = left_pos[:, 0]
    left_y = left_pos[:, 1]
    left_z = left_pos[:, 2]

    left_done = left_x < left_max_x
    left_done = torch.logical_and(left_done, left_x > left_min_x)
    left_done = torch.logical_and(left_done, left_y < left_max_y)
    left_done = torch.logical_and(left_done, left_y > left_min_y)
    left_done = torch.logical_and(left_done, left_z < max_height)
    left_done = torch.logical_and(left_done, left_vel[:, 0] < max_vel)
    left_done = torch.logical_and(left_done, left_vel[:, 1] < max_vel)
    left_done = torch.logical_and(left_done, left_vel[:, 2] < max_vel)

    # --- Right cube ---
    right_object: RigidObject = env.scene[right_object_cfg.name]
    right_pos = right_object.data.root_pos_w - env.scene.env_origins
    right_vel = torch.abs(right_object.data.root_vel_w)

    right_x = right_pos[:, 0]
    right_y = right_pos[:, 1]
    right_z = right_pos[:, 2]

    right_done = right_x < right_max_x
    right_done = torch.logical_and(right_done, right_x > right_min_x)
    right_done = torch.logical_and(right_done, right_y < right_max_y)
    right_done = torch.logical_and(right_done, right_y > right_min_y)
    right_done = torch.logical_and(right_done, right_z < max_height)
    right_done = torch.logical_and(right_done, right_vel[:, 0] < max_vel)
    right_done = torch.logical_and(right_done, right_vel[:, 1] < max_vel)
    right_done = torch.logical_and(right_done, right_vel[:, 2] < max_vel)

    # success only if BOTH cubes satisfy their conditions
    done = torch.logical_and(left_done, right_done)
    return done
