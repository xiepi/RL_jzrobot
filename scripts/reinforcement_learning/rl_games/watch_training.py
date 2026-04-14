"""Continuously watch a training run and periodically evaluate new checkpoints."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_LOG_ROOT = Path(r"E:\isaac-lab\IsaacLab\logs\rl_games\jz_bi_reach")
DEFAULT_TAGS = (
    "rewards/iter",
    "shaped_rewards/iter",
    "Episode/Metrics/left_ee_pose/position_error",
    "Episode/Metrics/right_ee_pose/position_error",
    "Episode/Episode_Reward/action_rate",
    "Episode/Episode_Reward/action_max_abs_penalty",
    "Episode/Episode_Reward/left_joint_vel",
    "Episode/Episode_Reward/right_joint_vel",
    "Episode/Episode_Reward/left_end_effector_stable_goal_bonus",
    "Episode/Episode_Reward/right_end_effector_stable_goal_bonus",
    "Episode/Episode_Reward/left_tcp_speed_near_goal",
    "Episode/Episode_Reward/right_tcp_speed_near_goal",
    "Episode/Episode_Reward/left_action_rate_near_goal",
    "Episode/Episode_Reward/right_action_rate_near_goal",
)
CHECKPOINT_PATTERN = re.compile(r"last_jz_bi_reach_ep_(\d+)_rew_.*\.pth$")


@dataclass
class ScalarPoint:
    step: int
    value: float


def _latest_event_file(run_dir: Path) -> Path | None:
    events = sorted((run_dir / "summaries").glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return events[0] if events else None


def _load_scalars(event_file: Path) -> tuple[event_accumulator.EventAccumulator, dict[str, ScalarPoint]]:
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalars: dict[str, ScalarPoint] = {}
    for tag in DEFAULT_TAGS:
        if tag not in ea.Tags().get("scalars", []):
            continue
        values = ea.Scalars(tag)
        if not values:
            continue
        scalars[tag] = ScalarPoint(step=values[-1].step, value=values[-1].value)
    return ea, scalars


def _format_scalar(name: str, point: ScalarPoint | None) -> str:
    if point is None:
        return f"{name}=NA"
    return f"{name}={point.value:.6f}@{point.step}"


def _append(log_file: Path, text: str) -> None:
    with log_file.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def _latest_checkpoint(nn_dir: Path) -> tuple[Path | None, int | None]:
    latest_path = None
    latest_epoch = None
    for path in sorted(nn_dir.glob("last_jz_bi_reach_ep_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
        match = CHECKPOINT_PATTERN.match(path.name)
        if not match:
            continue
        latest_path = path
        latest_epoch = int(match.group(1))
        break
    return latest_path, latest_epoch


def _run_eval(
    eval_script: Path,
    checkpoint_path: Path,
    task: str,
    num_envs: int,
    steps: int,
    output_log: Path,
) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--task",
        task,
        "--num_envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--headless",
        "--checkpoint",
        str(checkpoint_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), check=False)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append(output_log, f"[{timestamp}] EVAL {checkpoint_path.name}")
    if result.stdout:
        _append(output_log, result.stdout.rstrip())
    if result.stderr:
        _append(output_log, result.stderr.rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch training progress and evaluate new checkpoints.")
    parser.add_argument("--run", required=True, help="Run directory name under the RL-Games log root.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--task", type=str, default="Isaac-Reach-JZ-Bi-v0")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluate checkpoints every N epochs.")
    parser.add_argument("--eval-start", type=int, default=200, help="First epoch eligible for evaluation.")
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=80)
    args = parser.parse_args()

    run_dir = args.log_root / args.run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    status_log = run_dir / "watch_status.log"
    eval_log = run_dir / "watch_eval.log"
    eval_script = Path(__file__).with_name("evaluate_checkpoint.py")

    seen_eval_epochs: set[int] = set()
    last_event_mtime = None

    _append(status_log, f"[{datetime.now():%Y-%m-%d %H:%M:%S}] WATCH_START run={args.run}")

    while True:
        event_file = _latest_event_file(run_dir)
        if event_file is not None:
            event_mtime = event_file.stat().st_mtime
            if last_event_mtime != event_mtime:
                _, scalars = _load_scalars(event_file)
                summary = " ".join(
                    [
                        _format_scalar("reward", scalars.get("rewards/iter")),
                        _format_scalar("left_err", scalars.get("Episode/Metrics/left_ee_pose/position_error")),
                        _format_scalar("right_err", scalars.get("Episode/Metrics/right_ee_pose/position_error")),
                        _format_scalar("action_rate", scalars.get("Episode/Episode_Reward/action_rate")),
                        _format_scalar("action_max", scalars.get("Episode/Episode_Reward/action_max_abs_penalty")),
                        _format_scalar(
                            "stable_l", scalars.get("Episode/Episode_Reward/left_end_effector_stable_goal_bonus")
                        ),
                        _format_scalar(
                            "stable_r", scalars.get("Episode/Episode_Reward/right_end_effector_stable_goal_bonus")
                        ),
                    ]
                )
                _append(status_log, f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {summary}")
                last_event_mtime = event_mtime

        checkpoint_path, checkpoint_epoch = _latest_checkpoint(run_dir / "nn")
        if (
            checkpoint_path is not None
            and checkpoint_epoch is not None
            and checkpoint_epoch >= args.eval_start
            and checkpoint_epoch % args.eval_every == 0
            and checkpoint_epoch not in seen_eval_epochs
        ):
            _run_eval(eval_script, checkpoint_path, args.task, args.eval_num_envs, args.eval_steps, eval_log)
            seen_eval_epochs.add(checkpoint_epoch)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
