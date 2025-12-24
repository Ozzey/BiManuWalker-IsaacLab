# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-stack state machine for the stack cube env with an explicit orientation state.

Task:
    In each env:
      1) Move above cube_2,
      2) Re-orient the gripper there,
      3) Descend and grasp cube_2,
      4) Lift it,
      5) Move above cube_1,
      6) Place cube_2 centered on top of cube_1,
      7) Open gripper,
      8) Retreat straight up,
      9) Done.

Environment:
    Gym ID: "Isaac-Stack-Cube-Franka-IK-Abs-v0"
    Action: [x, y, z, qw, qx, qy, qz, gripper]
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick and stack state machine for stack cube environment.")
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

# Hover offset above cube center for approach phases
ABOVE_OFFSET_Z = 0.10  # [m]

# Distance threshold for reaching poses
POSITION_THRESHOLD = 0.13  # [m]

# Yaw offset between cube_2 yaw and gripper yaw
YAW_OFFSET_RAD = 0.0  # try 0.785398 for 45°

# Size of cubes (center-to-center offset in Z when stacked)
CUBE_HEIGHT = 0.1  # [m] (you tuned this already)

# Extra clearance so we *don't* push the cube into cube_1 when placing
STACK_CLEARANCE_Z = 0.01  # [m] small positive value

# Shift of the TCP relative to cube_2 center in its local +X direction (for picking only)
GRASP_FORWARD_OFFSET_LOCAL = -0.02  # [m]

# Wait times (seconds) per state
REST_WAIT = 0.2
APPROACH_ABOVE_WAIT = 0.5
ORIENT_ABOVE_WAIT = 0.4
APPROACH_OBJECT_WAIT = 0.6
GRASP_WAIT = 0.3
LIFT_WAIT = 0.7
MOVE_ABOVE_STACK_WAIT = 0.7
PLACE_ON_STACK_WAIT = 0.5
RELEASE_WAIT = 0.3
RETREAT_UP_WAIT = 0.3


RELEASE_OBJECT_STATE = 8
# =======================
# Warp setup
# =======================

wp.init()


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class StackSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    ORIENT_ABOVE_OBJECT = wp.constant(2)
    APPROACH_OBJECT = wp.constant(3)
    GRASP_OBJECT = wp.constant(4)
    LIFT_OBJECT = wp.constant(5)
    MOVE_ABOVE_STACK = wp.constant(6)
    PLACE_ON_STACK = wp.constant(7)
    RELEASE_OBJECT = wp.constant(8)
    RETREAT_UP = wp.constant(9)
    DONE = wp.constant(10)


class StackSmWaitTime:
    REST = wp.constant(REST_WAIT)
    APPROACH_ABOVE_OBJECT = wp.constant(APPROACH_ABOVE_WAIT)
    ORIENT_ABOVE_OBJECT = wp.constant(ORIENT_ABOVE_WAIT)
    APPROACH_OBJECT = wp.constant(APPROACH_OBJECT_WAIT)
    GRASP_OBJECT = wp.constant(GRASP_WAIT)
    LIFT_OBJECT = wp.constant(LIFT_WAIT)
    MOVE_ABOVE_STACK = wp.constant(MOVE_ABOVE_STACK_WAIT)
    PLACE_ON_STACK = wp.constant(PLACE_ON_STACK_WAIT)
    RELEASE_OBJECT = wp.constant(RELEASE_WAIT)
    RETREAT_UP = wp.constant(RETREAT_UP_WAIT)
    DONE = wp.constant(0.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_stack_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    stack_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    # Above-object and above-stack poses
    above_object = wp.transform_multiply(offset[tid], object_pose[tid])
    above_stack = wp.transform_multiply(offset[tid], stack_pose[tid])

    above_object_pos = wp.transform_get_translation(above_object)
    above_stack_pos = wp.transform_get_translation(above_stack)
    object_rot = wp.transform_get_rotation(object_pose[tid])

    if state == StackSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= StackSmWaitTime.REST:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == StackSmState.APPROACH_ABOVE_OBJECT:
        # Move above cube_2, keep current orientation
        cur_rot = wp.transform_get_rotation(ee_pose[tid])
        des_ee_pose[tid] = wp.transform(above_object_pos, cur_rot)
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_object_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = StackSmState.ORIENT_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.ORIENT_ABOVE_OBJECT:
        # Stay above cube_2, rotate to its chosen orientation
        des_ee_pose[tid] = wp.transform(above_object_pos, object_rot)
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_object_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.ORIENT_ABOVE_OBJECT:
                sm_state[tid] = StackSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.APPROACH_OBJECT:
        # Descend to cube_2
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = StackSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        if sm_wait_time[tid] >= StackSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = StackSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == StackSmState.LIFT_OBJECT:
        # Lift cube_2 to its "above-object" pose
        des_ee_pose[tid] = above_object
        gripper_state[tid] = GripperState.CLOSE

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(above_object),
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = StackSmState.MOVE_ABOVE_STACK
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.MOVE_ABOVE_STACK:
        # Move above the target stack pose
        des_ee_pose[tid] = above_stack
        gripper_state[tid] = GripperState.CLOSE

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_stack_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.MOVE_ABOVE_STACK:
                sm_state[tid] = StackSmState.PLACE_ON_STACK
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.PLACE_ON_STACK:
        # Descend to stack pose (centered on cube_1, with clearance)
        des_ee_pose[tid] = stack_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(stack_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.PLACE_ON_STACK:
                sm_state[tid] = StackSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = stack_pose[tid]
        gripper_state[tid] = GripperState.OPEN

        if sm_wait_time[tid] >= StackSmWaitTime.RELEASE_OBJECT:
            # New: retreat vertically after releasing
            sm_state[tid] = StackSmState.RETREAT_UP
            sm_wait_time[tid] = 0.0

    elif state == StackSmState.RETREAT_UP:
        # Move straight up to above_stack (same XY, higher Z)
        des_ee_pose[tid] = above_stack
        gripper_state[tid] = GripperState.OPEN

        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            above_stack_pos,
            position_threshold,
        ):
            if sm_wait_time[tid] >= StackSmWaitTime.RETREAT_UP:
                sm_state[tid] = StackSmState.DONE
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.DONE:
        # Hold pose, keep gripper open
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndStackSm:
    """State machine to pick cube_2 and stack it onto cube_1."""

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

        # Hover offset: only z translation (orientation identity)
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
        ee_pose: torch.Tensor,      # [N, 7] (x, y, z, qw, qx, qy, qz)
        object_pose: torch.Tensor,  # [N, 7]
        stack_pose: torch.Tensor,   # [N, 7]
    ) -> torch.Tensor:
        # convert to (x, y, z, x, y, z, w) for warp
        ee_pose_wp_fmt = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose_wp_fmt = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        stack_pose_wp_fmt = stack_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose_wp_fmt.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose_wp_fmt.contiguous(), wp.transform)
        stack_pose_wp = wp.from_torch(stack_pose_wp_fmt.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_stack_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                stack_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        wp.synchronize()

        # back to (x, y, z, qw, qx, qy, qz)
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

    # action buffer (position + quaternion + gripper)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0  # qw

    stack_sm = PickAndStackSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=POSITION_THRESHOLD,
    )

    # --- New: per-env frozen orientation after stacking ---
    frozen_orientation = torch.zeros(
        (env.unwrapped.num_envs, 4), device=env.unwrapped.device
    )
    frozen_orientation[:, 0] = 1.0

    while simulation_app.is_running():
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            cube1_data: RigidObjectData = env.unwrapped.scene["cube_1"].data
            cube2_data: RigidObjectData = env.unwrapped.scene["cube_2"].data

            cube1_pos = cube1_data.root_pos_w - env.unwrapped.scene.env_origins
            cube2_pos = cube2_data.root_pos_w - env.unwrapped.scene.env_origins
            cube2_quat = cube2_data.root_quat_w  # (w,x,y,z)

            # --- New: use state machine state to decide if orientation should be updated ---
            sm_state = stack_sm.sm_state          # [N], int32
            mask_update = sm_state < RELEASE_OBJECT_STATE  # update only before RELEASE_OBJECT

            cube2_yaw = yaw_from_quat(cube2_quat)
            yaw = cube2_yaw + YAW_OFFSET_RAD

            yaw_quat = quat_from_yaw(yaw)    # Rz(yaw)
            base_down = torch.zeros_like(cube2_quat)
            base_down[:, 1] = 1.0           # (0,1,0,0) = Rx(pi)
            desired_orientation_all = quat_mul(yaw_quat, base_down)  # (w,x,y,z)

            # Start from the frozen orientation and update only those envs that are not yet in RELEASE+
            desired_orientation = frozen_orientation.clone()
            desired_orientation[mask_update] = desired_orientation_all[mask_update]
            frozen_orientation[mask_update] = desired_orientation_all[mask_update]
            # --- End new orientation logic ---

            # --- Position offset along cube_2 local X for picking ---
            dx = GRASP_FORWARD_OFFSET_LOCAL
            world_dx = torch.cos(cube2_yaw) * dx
            world_dy = torch.sin(cube2_yaw) * dx

            # TCP target for picking cube_2 (use offset)
            object_pos = cube2_pos.clone()
            object_pos[:, 0] += world_dx
            object_pos[:, 1] += world_dy

            # TCP target for stacking ABOVE cube_1 center (no XY offset, but with clearance in Z)
            stack_pos = cube1_pos.clone()
            stack_pos[:, 2] += CUBE_HEIGHT + STACK_CLEARANCE_Z

            ee_pose     = torch.cat([tcp_pos, tcp_quat], dim=-1)
            object_pose = torch.cat([object_pos, desired_orientation], dim=-1)
            stack_pose  = torch.cat([stack_pos,  desired_orientation], dim=-1)

            actions = stack_sm.compute(ee_pose, object_pose, stack_pose)


            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                stack_sm.reset_idx(done_ids)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
