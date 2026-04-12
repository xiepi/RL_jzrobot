"""Custom reward helpers for JZ synthetic TCP tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_batch_quat(env: ManagerBasedRLEnv, quat: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor(quat, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)


def _base_link_index(asset: RigidObject) -> int:
    return asset.body_names.index("base_link")


def _base_link_pose_w(asset: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    base_idx = _base_link_index(asset)
    return asset.data.body_pos_w[:, base_idx], asset.data.body_quat_w[:, base_idx]


def _fingertip_midpoint_pos_w(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)  # type: ignore[index]


def fingertip_midpoint_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Synthetic TCP position of the gripper center in the robot base_link frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos_w, base_quat_w = _base_link_pose_w(asset)
    curr_pos_w = _fingertip_midpoint_pos_w(asset, asset_cfg)
    curr_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, curr_pos_w)
    return curr_pos_b


def fingertip_midpoint_position_command_error_vector_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Desired minus current gripper-center position in the base_link frame."""

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    curr_pos_b = fingertip_midpoint_position_b(env, asset_cfg)
    return des_pos_b - curr_pos_b


def generated_command_positions(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Desired command position in the base_link frame."""

    return env.command_manager.get_command(command_name)[:, :3]


def fingertip_midpoint_position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 position error between the desired TCP command and fingertip midpoint."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    base_pos_w, base_quat_w = _base_link_pose_w(asset)
    des_pos_w, _ = combine_frame_transforms(base_pos_w, base_quat_w, des_pos_b)
    curr_pos_w = _fingertip_midpoint_pos_w(asset, asset_cfg)
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def fingertip_midpoint_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tanh-shaped position reward using the fingertip midpoint as synthetic TCP."""

    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def joint_vel_l2_when_close_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    tcp_asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Joint-velocity penalty activated only when the gripper center is already close to the target."""

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = fingertip_midpoint_position_command_error(env, command_name, tcp_asset_cfg)
    is_close = (distance <= threshold).to(asset.data.joint_vel.dtype)
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return is_close * joint_vel_sq


def action_max_abs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the largest absolute action component to suppress single-joint spikes."""

    return torch.max(torch.abs(env.action_manager.action), dim=1).values


def orientation_command_error_with_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    offset_quat: tuple[float, float, float, float],
) -> torch.Tensor:
    """Orientation error against a body frame composed with a fixed local quaternion offset."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    _, base_quat_w = _base_link_pose_w(asset)
    des_quat_w = quat_mul(base_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore[index]
    curr_quat_w = quat_mul(curr_quat_w, _as_batch_quat(env, offset_quat))
    return quat_error_magnitude(curr_quat_w, des_quat_w)
