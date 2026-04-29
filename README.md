# JZ Isaac Lab

这是一个面向 Isaac Lab 的 JZ 双臂强化学习项目（当前主任务：双臂 Reach）。

## 项目定位

- `jz_isaac_lab`：强化学习任务定义、训练/回放脚本、监控与评估脚本。
- `jz_descripetion`：机器人 URDF 与网格（运行本项目必需）。
- `robot_simulator`：仿真与 IK 工具（生成 reachable workspace 时需要）。

当前任务：

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

## 30 分钟快速上手（Windows / PowerShell）

### 1) 建议目录结构

```text
<workspace_root>/
  IsaacLab/
  RL_jzrobot/
  jz_descripetion/
  robot_simulator/      # 仅 workspace 采样工具需要
```

### 2) 设置环境变量

```powershell
$env:ISAACLAB_PATH = "E:\isaac-lab\IsaacLab"
$env:JZLAB_WORKSPACE_ROOT = "E:\jz_robot"
$env:JZLAB_PROJECT_PATH = "$env:JZLAB_WORKSPACE_ROOT\RL_jzrobot"
```

说明：

- `ISAACLAB_PATH` 必填，用于调用 `isaaclab.bat`。
- `JZLAB_WORKSPACE_ROOT` 建议设置；代码会据此寻找 `jz_descripetion` 与 `robot_simulator`。

### 3) 安装 Python 包

```powershell
cd $env:JZLAB_PROJECT_PATH
python -m pip install -e .\source\jzlab
```

### 4) 验证任务已注册

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\list_envs.py"
```

看到 `Isaac-Reach-JZ-Bi-v0` / `Isaac-Reach-JZ-Bi-Play-v0` 即通过。

### 5) 生成 USD（首次建议执行）

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\convert_jz_bimanual.py" --headless
```

### 6) 训练冒烟（快速验证交接环境）

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 32 --max_iterations 5 --headless
```

日志目录：`$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>`

### 7) 回放 checkpoint

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\jz_bi_reach.pth"
```

## 一键训练（含自动 watcher）

```powershell
cd $env:JZLAB_PROJECT_PATH
powershell -ExecutionPolicy Bypass -File ".\scripts\reinforcement_learning\rl_games\launch_training_windows.ps1" -NumEnvs 1024 -MaxIterations 2000 -StartTensorBoard
```

可选参数：

- `-IsaacLabRoot`：显式指定 Isaac Lab 目录。
- `-CondaEnvName`：默认 `env_isaacsim`。
- `-RunName`：指定实验名称，便于交接追踪。

## 同事交接建议

建议先执行“训练冒烟 + checkpoint 回放”两项验收再正式交接。

详细清单见：`docs/handoff_quickstart_zh.md`

## 常见问题

- 找不到 `jz_descripetion`：检查 `JZLAB_WORKSPACE_ROOT` 是否正确。
- `ISAACLAB_PATH` 为空：先设置环境变量，或在启动脚本传 `-IsaacLabRoot`。
- 首次训练慢：USD 转换与 shader 缓存会拉长首轮耗时。

## 许可证

MIT，详见 `LICENSE`。
