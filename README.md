# JZ Isaac Lab

这是一个面向 Isaac Lab 的 JZ 双臂强化学习项目。

本仓库与以下模块解耦维护：

- `robot_simulator/`：Action Graph、ROS2、IK 与运行时验证
- `jz_descripetion/`：权威 URDF 与网格模型

当前已提供的任务：

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

当前版本中，机身与夹爪保持冻结，仅训练左右手臂关节位置策略（14-DoF 动作空间）。

## 环境要求

- 已正确安装并可运行 Isaac Lab。
- 已设置环境变量 `ISAACLAB_PATH`（指向 Isaac Lab 根目录）。
- 可选：设置 `JZLAB_PROJECT_PATH` 指向本仓库根目录；若不设置，示例默认使用当前目录。

## 安装

```bash
cd source/jzlab
python -m pip install -e .
```

## 生成 USD

```powershell
$env:JZLAB_PROJECT_PATH = (Get-Location).Path
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\convert_jz_bimanual.py" --headless
```

## 开始训练

```powershell
$env:JZLAB_PROJECT_PATH = (Get-Location).Path
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --headless
```

Windows 一键启动脚本：

```powershell
$env:JZLAB_PROJECT_PATH = (Get-Location).Path
powershell -ExecutionPolicy Bypass -File ".\scripts\reinforcement_learning\rl_games\launch_training_windows.ps1"
```

## 回放策略

```powershell
$env:JZLAB_PROJECT_PATH = (Get-Location).Path
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run>\nn\jz_bi_reach.pth"
```

## 许可证

MIT，详见 `LICENSE`。
