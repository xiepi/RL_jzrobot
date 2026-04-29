# JZ Isaac Lab

这是一个面向 Isaac Lab 的 JZ 双臂强化学习项目（当前主任务：双臂 Reach）。

## 项目定位

- `jz_isaac_lab`：强化学习任务定义、训练/回放脚本、监控与评估脚本。
- `jz_descripetion`：机器人 URDF 与网格（运行本项目必需）。
- `robot_simulator`：仿真与 IK 工具（生成 reachable workspace 时需要）。

当前任务：

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

## 项目结构（仓库内）

```text
RL_jzrobot/
  README.md
  docs/
    handoff_quickstart_zh.md               # 交接清单（中文）
  scripts/
    reinforcement_learning/rl_games/
      train.py                             # 训练入口
      play.py                              # 回放入口
      evaluate_checkpoint.py               # 离线评估
      monitor_training.py                  # 标量摘要
      watch_training.py                    # 训练期周期评估
      launch_training_windows.ps1          # Windows 一键训练
    tools/
      convert_jz_bimanual.py               # URDF -> USD
      list_envs.py                         # 列出 JZ 任务
      generate_reachable_workspace.py      # 采样可达空间数据
  source/jzlab/
    jzlab/tasks/manager_based/jz_manipulation/
      assets/                              # 机器人资产配置
      bimanual/reach/                      # Reach 任务配置与 MDP
      usds/                                # 生成的 USD 与配置
```

## 项目结构（工作区内）

本仓库依赖同级目录中的机器人描述仓库与（可选）仿真工具仓库：

```text
<workspace_root>/
  IsaacLab/                                # Isaac Lab 主仓库
  RL_jzrobot/                              # 当前仓库
  jz_descripetion/                         # 必需：URDF + meshes
  robot_simulator/                         # 可选：workspace 采样工具依赖
```

其中：

- `jz_descripetion` 是训练运行必需依赖。
- `robot_simulator` 在运行 `generate_reachable_workspace.py` 时需要。

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

### 2.1) 本机 IsaacLab 设置示例（LEGION）

以下是你当前机器上的已验证配置（2026-04-29）：

```powershell
cd E:\isaac-lab\IsaacLab
conda activate env_isaacsim
$env:ISAACLAB_PATH = "E:\isaac-lab\IsaacLab"
$env:JZLAB_WORKSPACE_ROOT = "E:\jz_robot"
$env:JZLAB_PROJECT_PATH = "E:\jz_robot\jz_isaac_lab"
```

已实测可启动训练的命令：

```powershell
.\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --headless --num_envs 64 --max_iterations 6000
```

说明：该次运行在 `epoch 9` 手动停止，验证了训练链路可用；若要回放模型，请按下文“checkpoint 回放”步骤先确保产出 `.pth` 文件。

### 2.2) `IsaacLab` 目录需要改什么（重点）

结论：通常**不需要修改 IsaacLab 源码**，只需要做运行环境层面的最小设置。

必须项：

1. 使用 IsaacLab 对应的 conda 环境（例如 `env_isaacsim`）。
2. 设置 `ISAACLAB_PATH` 指向 IsaacLab 根目录。
3. 把本仓库包安装到该环境：`python -m pip install -e .\source\jzlab`。

可选项（为了每次终端自动生效）：

```powershell
setx ISAACLAB_PATH "E:\isaac-lab\IsaacLab"
setx JZLAB_WORKSPACE_ROOT "E:\jz_robot"
setx JZLAB_PROJECT_PATH "E:\jz_robot\jz_isaac_lab"
```

不建议做的事情：

- 不建议直接改 `IsaacLab` 仓库里的任务源码来适配本项目。
- 不建议把 `jzlab` 代码复制进 `IsaacLab/source`（后续升级难维护）。
- 不建议硬编码同事机器路径（优先使用环境变量）。

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

若要保证产出可回放模型（checkpoint），建议至少执行：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

然后检查：

```powershell
Get-ChildItem "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\*.pth"
```

### 7) 回放 checkpoint

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\jz_bi_reach.pth"
```

## 一键训练（含自动监控）

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
