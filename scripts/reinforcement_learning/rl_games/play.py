"""Play back RL-Games checkpoints for JZ Isaac Lab tasks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PACKAGE_DIR = PROJECT_ROOT / "source" / "jzlab"
if str(SOURCE_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_DIR))

ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "")).resolve() if os.environ.get("ISAACLAB_PATH") else None
if ISAACLAB_ROOT:
    for rel_path in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_rl", "source/isaaclab_assets"):
        candidate = ISAACLAB_ROOT / rel_path
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published pre-trained checkpoint.")
parser.add_argument(
    "--use_last_checkpoint", action="store_true", help="Use the last saved model if no checkpoint is provided."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import math
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import jzlab.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve_play_task_name(task_name: str | None) -> str | None:
    """Prefer the dedicated *-Play-vN task variant for playback when available."""

    if task_name is None or "-Play-" in task_name:
        return task_name

    if "-v" not in task_name:
        return task_name

    stem, version = task_name.rsplit("-v", 1)
    candidate = f"{stem}-Play-v{version}"
    try:
        gym.spec(candidate)
    except Exception:
        return task_name

    print(f"[INFO] Switching playback task from '{task_name}' to dedicated play task '{candidate}'.")
    return candidate


args_cli.task = _resolve_play_task_name(args_cli.task)


def _load_checkpoint_file(checkpoint_path: str) -> dict:
    """Load a trusted local checkpoint across PyTorch versions."""

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _infer_checkpoint_obs_dim(checkpoint_path: str) -> int | None:
    """Infer the policy observation dimension stored in an RL-Games checkpoint."""

    checkpoint = _load_checkpoint_file(checkpoint_path)
    model_state = checkpoint.get("model", {})

    running_mean = model_state.get("running_mean_std.running_mean")
    if running_mean is not None and len(running_mean.shape) == 1:
        return int(running_mean.shape[0])

    actor_weight = model_state.get("a2c_network.actor_mlp.0.weight")
    if actor_weight is not None and len(actor_weight.shape) == 2:
        return int(actor_weight.shape[1])

    return None


def _infer_checkpoint_mlp_units(checkpoint_path: str) -> list[int] | None:
    """Infer hidden-layer widths from an RL-Games actor MLP checkpoint."""

    checkpoint = _load_checkpoint_file(checkpoint_path)
    model_state = checkpoint.get("model", {})

    units: list[int] = []
    layer_index = 0
    while True:
        weight = model_state.get(f"a2c_network.actor_mlp.{layer_index}.weight")
        if weight is None:
            break
        if len(weight.shape) != 2:
            break
        units.append(int(weight.shape[0]))
        layer_index += 2

    return units or None


def _apply_network_compat_for_checkpoint(agent_cfg: dict, checkpoint_path: str) -> bool:
    """Align the agent MLP layout with the checkpoint before building the policy."""

    checkpoint_units = _infer_checkpoint_mlp_units(checkpoint_path)
    if checkpoint_units is None:
        return False

    network_cfg = agent_cfg.get("params", {}).get("network", {})
    mlp_cfg = network_cfg.get("mlp", {})
    current_units = list(mlp_cfg.get("units", []))
    if current_units == checkpoint_units:
        return False

    mlp_cfg["units"] = checkpoint_units
    print(
        "[INFO] Applying checkpoint network compatibility for playback: "
        f"using hidden units {checkpoint_units} instead of configured {current_units}."
    )
    return True


def _apply_obs_compat_for_checkpoint(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, obs_dim: int) -> bool:
    """Adjust known observation-layout changes so older checkpoints can still be played back."""

    policy_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_cfg is None:
        return False

    if obs_dim == 56:
        changed = False
        for term_name in ("left_pose_command", "right_pose_command"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                getattr(policy_cfg, term_name).func = jzlab.tasks.manager_based.jz_manipulation.bimanual.reach.mdp.generated_commands
        for term_name in ("left_tcp_error", "right_tcp_error"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                setattr(policy_cfg, term_name, None)
                changed = True
        # Match the pre-smoothing training configuration for older checkpoints.
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
            env_cfg.sim.physx.enable_external_forces_every_iteration = False
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        if robot_cfg is not None and getattr(robot_cfg, "spawn", None) is not None:
            articulation_props = getattr(robot_cfg.spawn, "articulation_props", None)
            if articulation_props is not None:
                articulation_props.solver_velocity_iteration_count = 1
        if robot_cfg is not None and getattr(robot_cfg, "actuators", None) is not None:
            arm_actuator = robot_cfg.actuators.get("arm")
            if arm_actuator is not None:
                arm_actuator.damping = 80.0
        if changed:
            print("[INFO] Applying legacy 56-dim checkpoint compatibility for playback.")
        return changed

    if obs_dim == 62:
        changed = False
        for term_name in ("left_pose_command", "right_pose_command"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                getattr(policy_cfg, term_name).func = jzlab.tasks.manager_based.jz_manipulation.bimanual.reach.mdp.generated_commands
                changed = True
        if changed:
            print("[INFO] Applying legacy 62-dim checkpoint compatibility for playback.")
        return changed

    return False


def _infer_env_policy_obs_dim(env: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | gym.Env) -> int | None:
    """Infer the unbatched policy observation dimension from an Isaac Lab env instance."""

    base_env = getattr(env, "unwrapped", env)
    single_obs_space = getattr(base_env, "single_observation_space", None)
    if single_obs_space is not None:
        policy_space = single_obs_space.get("policy")
        if policy_space is not None and getattr(policy_space, "shape", None) is not None:
            return int(policy_space.shape[0])

    obs_space = getattr(env, "observation_space", None)
    if getattr(obs_space, "shape", None) is not None:
        return int(obs_space.shape[0])

    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint is currently available for this task.")
            return
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env_cfg.log_dir = log_dir
    checkpoint_obs_dim = _infer_checkpoint_obs_dim(resume_path)
    if checkpoint_obs_dim is not None:
        _apply_obs_compat_for_checkpoint(env_cfg, checkpoint_obs_dim)
    _apply_network_compat_for_checkpoint(agent_cfg, resume_path)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if checkpoint_obs_dim is not None:
        env_obs_dim = _infer_env_policy_obs_dim(env)
        if env_obs_dim is None:
            raise RuntimeError("Unable to infer policy observation dimension from the constructed environment.")
        if env_obs_dim != checkpoint_obs_dim:
            raise RuntimeError(
                "Checkpoint observation dimension mismatch after compatibility handling: "
                f"checkpoint expects {checkpoint_obs_dim}, environment provides {env_obs_dim}. "
                "Use a checkpoint trained with the current observation layout or add another compatibility mapping."
            )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
