# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-lift state machine for the stack cube env with an explicit orientation state.

Task:
    In each env, move above cube_1, re-orient the gripper there, descend to grasp,
    close the gripper, and lift the cube straight up by LIFT_HEIGHT.

Environment:
    Gym ID: "Isaac-Stack-Cube-Franka-IK-Abs-v0"
    Action: [x, y, z, qw, qx, qy, qz, gripper]
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick and lift state machine for stack cube environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# =======================
# Tunable parameters
# =======================

# Hover offset above cube center for approach and after lift
ABOVE_OFFSET_Z = 0.10  # [m]

# Vertical lift relative to cube center
LIFT_HEIGHT = 0.15  # [m]

# Distance threshold for reaching poses
POSITION_THRESHOLD = 0.13  # [m]

# Yaw offset between cube yaw and gripper yaw (to avoid edge collisions)
#   0.0      -> align fingers with cube faces
#   pi/4     -> fingers at 45° relative to cube edges
YAW_OFFSET_RAD = 0.0  # try 0.785398 for 45°

# Wait times (seconds) per state
REST_WAIT = 0.2
APPROACH_ABOVE_WAIT = 0.5
ORIENT_WAIT = 0.4
APPROACH_OBJECT_WAIT = 0.6
GRASP_WAIT = 0.3
LIFT_WAIT = 1.0


GRASP_FORWARD_OFFSET_LOCAL = -0.02

# =======================
# Warp setup
# =======================

wp.init()


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    ORIENT_ABOVE_OBJECT = wp.constant(2)
    APPROACH_OBJECT = wp.constant(3)
    GRASP_OBJECT = wp.constant(4)
    LIFT_OBJECT = wp.constant(5)


class PickSmWaitTime:
    REST = wp.constant(REST_WAIT)
    APPROACH_ABOVE_OBJECT = wp.constant(APPROACH_ABOVE_WAIT)
    ORIENT_ABOVE_OBJECT = wp.constant(ORIENT_WAIT)
    APPROACH_OBJECT = wp.constant(APPROACH_OBJECT_WAIT)
    GRASP_OBJECT = wp.constant(GRASP_WAIT)
    LIFT_OBJECT = wp.constant(LIFT_WAIT)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    # Above-object pose (position above object, orientation = object's orientation)
    above_object = wp.transform_multiply(offset[tid], object_pose[tid])
    above_pos = wp.transform_get_translation(above_object)
    object_rot = wp.transform_get_rotation(object_pose[tid])

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        # Move to a point above the object but keep current orientation
        cur_rot = wp.transform_get_rotation(ee_pose[tid])
        des_ee_pose[tid] = wp.transform(above_pos, cur_rot)
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.ORIENT_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.ORIENT_ABOVE_OBJECT:
        # Stay at the above-object position, rotate to object's orientation
        des_ee_pose[tid] = wp.transform(above_pos, object_rot)
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_pos,
            position_threshold,
        ):
            # we don't check orientation explicitly; time + stable IK is usually enough
            if sm_wait_time[tid] >= PickSmWaitTime.ORIENT_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_OBJECT:
        # Descend to object pose with the oriented gripper
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # final state: keep holding
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Lift-only state machine for cube_1 in the stack env, with explicit orientation state."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # hover offset: only z translation (orientation identity)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = ABOVE_OFFSET_Z
        self.offset[:, -1] = 1.0  # identity (x, y, z, w)

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,        # [N, 7] (x, y, z, qw, qx, qy, qz)
        object_pose: torch.Tensor,    # [N, 7]
        des_object_pose: torch.Tensor # [N, 7]
    ) -> torch.Tensor:
        ee_pose_wp_fmt = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose_wp_fmt = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose_wp_fmt = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose_wp_fmt.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose_wp_fmt.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose_wp_fmt.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        wp.synchronize()

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# =======================
# Quaternion helpers (Torch)
# =======================

def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) for rotation about world Z by yaw."""
    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    zero = torch.zeros_like(qw)
    return torch.stack([qw, zero, zero, qz], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication q = q1 * q2 (w,x,y,z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract world Z-yaw from (w,x,y,z) quaternion."""
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# =======================
# Main loop
# =======================

def main():
    TASK_NAME = "Isaac-Stack-Cube-Franka-IK-Abs-v0"

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0  # qw

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=POSITION_THRESHOLD,
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            # cube_1 is the object to pick
            cube2_data: RigidObjectData = env.unwrapped.scene["cube_1"].data
            object_pos = cube2_data.root_pos_w - env.unwrapped.scene.env_origins
            object_quat = cube2_data.root_quat_w  # (w,x,y,z)


            # Orientation: yaw-align gripper with cube + optional offset, then Rx(pi) for "down"
            cube_yaw = yaw_from_quat(object_quat)
            yaw = cube_yaw + YAW_OFFSET_RAD


            # ---- NEW: position offset along cube local X ----
            # local offset (dx, 0, 0) in cube frame
            dx = GRASP_FORWARD_OFFSET_LOCAL
            world_dx = torch.cos(yaw) * dx
            world_dy = torch.sin(yaw) * dx
            object_pos[:, 0] += world_dx
            object_pos[:, 1] += world_dy
            # -------------------------------------------------



            yaw_quat = quat_from_yaw(yaw)    # Rz(yaw)
            base_down = torch.zeros_like(object_quat)
            base_down[:, 1] = 1.0           # (0,1,0,0) = Rx(pi)
            desired_orientation = quat_mul(yaw_quat, base_down)  # (w,x,y,z)

            # Lift target
            desired_pos = object_pos.clone()
            desired_pos[:, 2] += LIFT_HEIGHT

            ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
            object_pose = torch.cat([object_pos, desired_orientation], dim=-1)
            des_object_pose = torch.cat([desired_pos, desired_orientation], dim=-1)

            actions = pick_sm.compute(ee_pose, object_pose, des_object_pose)

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                pick_sm.reset_idx(done_ids)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
