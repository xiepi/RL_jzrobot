"""Command terms that sample targets from precomputed reachable workspace datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_mul, quat_unique

from ..workspace_data import ensure_workspace_dataset


class ReachableWorkspacePoseCommand(UniformPoseCommand):
    """Sample base-frame pose commands from a precomputed workspace dataset."""

    cfg: "ReachableWorkspacePoseCommandCfg"

    def __init__(self, cfg: "ReachableWorkspacePoseCommandCfg", env):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        data_path = ensure_workspace_dataset(cfg.dataset_file)
        if not data_path.is_file():
            raise FileNotFoundError(f"Workspace dataset not found: {data_path}")

        raw_data = json.loads(data_path.read_text(encoding="utf-8"))
        positions = raw_data.get(cfg.dataset_key, [])
        if not positions:
            raise ValueError(f"Workspace dataset key '{cfg.dataset_key}' is empty in {data_path}")

        self._positions_b = torch.tensor(positions, device=self.device, dtype=torch.float32)
        self._fixed_quaternion: torch.Tensor | None = None
        if cfg.use_fixed_quaternion:
            fixed_quaternion = torch.tensor(cfg.fixed_quaternion, device=self.device, dtype=torch.float32).view(1, 4)
            self._fixed_quaternion = quat_unique(fixed_quaternion) if self.cfg.make_quat_unique else fixed_quaternion

        self._base_link_idx = self.robot.body_names.index("base_link")
        self._current_position_body_ids: list[int] | None = None
        if cfg.current_position_body_names:
            self._current_position_body_ids = list(self.robot.find_bodies(list(cfg.current_position_body_names))[0])

        self._current_quaternion_offset: torch.Tensor | None = None
        if cfg.current_quaternion_offset is not None:
            quat_offset = torch.tensor(cfg.current_quaternion_offset, device=self.device, dtype=torch.float32).view(1, 4)
            self._current_quaternion_offset = quat_offset.expand(self.num_envs, -1)

    def _base_link_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.body_pos_w[:, self._base_link_idx], self.robot.data.body_quat_w[:, self._base_link_idx]

    def _current_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_position_body_ids is None:
            curr_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        else:
            curr_pos_w = self.robot.data.body_pos_w[:, self._current_position_body_ids, :].mean(dim=1)

        curr_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        if self._current_quaternion_offset is not None:
            curr_quat_w = quat_mul(curr_quat_w, self._current_quaternion_offset)
        return curr_pos_w, curr_quat_w

    def _update_metrics(self):
        base_pos_w, base_quat_w = self._base_link_pose_w()
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            base_pos_w,
            base_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        curr_pos_w, curr_quat_w = self._current_pose_w()
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            curr_pos_w,
            curr_quat_w,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        sample_ids = torch.randint(0, self._positions_b.shape[0], (len(env_ids),), device=self.device)
        self.pose_command_b[env_ids, :3] = self._positions_b[sample_ids]

        if self._fixed_quaternion is not None:
            self.pose_command_b[env_ids, 3:] = self._fixed_quaternion.expand(len(env_ids), -1)
        else:
            euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
            euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        curr_pos_w, curr_quat_w = self._current_pose_w()
        self.current_pose_visualizer.visualize(curr_pos_w, curr_quat_w)


@configclass
class ReachableWorkspacePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for reachable-workspace pose sampling."""

    class_type: type = ReachableWorkspacePoseCommand

    dataset_file: str = ""
    """Path to a JSON workspace dataset file."""

    dataset_key: str = ""
    """JSON key containing an array of base-frame XYZ positions."""

    use_fixed_quaternion: bool = False
    """Whether to use a fixed quaternion instead of sampling Euler angle ranges."""

    fixed_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Fixed target orientation quaternion in (w, x, y, z), expressed in base_link frame."""

    current_position_body_names: tuple[str, ...] = ()
    """Bodies used to visualize and score the current TCP position. If empty, uses ``body_name``."""

    current_quaternion_offset: tuple[float, float, float, float] | None = None
    """Optional quaternion offset composed onto ``body_name`` for the current TCP orientation."""

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (0.0, 0.0)
        pos_y: tuple[float, float] = (0.0, 0.0)
        pos_z: tuple[float, float] = (0.0, 0.0)
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()
