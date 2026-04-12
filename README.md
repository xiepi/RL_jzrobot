# JZ Isaac Lab

JZ dual-arm reinforcement learning package for Isaac Lab.

This project is intentionally separate from:

- `robot_simulator/` for Action Graph, ROS2, IK, and runtime validation
- `jz_descripetion/` for the authoritative URDF and meshes

The first task in this project is:

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

This initial version freezes the body and grippers and trains only the left/right arm
joint-position policy with a 14-DoF action space.

## Install

```bash
cd E:/jz_robot/jz_isaac_lab/source/jzlab
python -m pip install -e .
```

## Generate USD

```powershell
cd E:\isaac-lab\IsaacLab
.\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\tools\convert_jz_bimanual.py" --headless
```

## Train

```powershell
cd E:\isaac-lab\IsaacLab
.\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --headless
```

## Play

```powershell
cd E:\isaac-lab\IsaacLab
.\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "E:\isaac-lab\IsaacLab\logs\rl_games\jz_bi_reach\<run>\nn\jz_bi_reach.pth"
```
