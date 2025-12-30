# MountainCar-v0: Baseline DQN vs Trajectory Densification

这个小项目用于实践Coreset在DQN中的运用：
- **Baseline**: 标准 DQN + 经验回放 + 均匀采样
- **Ours**: 基于关键性度量的核心集缓冲区 (Dcore) + 时序重连 (temporal reconnection) 的 densified transitions

## 安装

```bash
cd /Users/username/Desktop/信息论
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练

Baseline:
```bash
python -m mc_dqn.train --variant baseline --episodes 800 --seed 0
```

Ours:
```bash
python -m mc_dqn.train --variant ours --episodes 800 --seed 0
```

日志会写入 `runs/<variant>_seed<seed>/metrics.csv`。

## 画图 (Figure 1)

```bash
python -m mc_dqn.plot_results --run_dir runs/baseline_seed0 --run_dir runs/ours_seed0
```

会输出 `runs/compare_reward.png`。

## 样本效率统计（论文指标）

导出“达到相同性能指标（最近 100 回合平均奖励 > -150）所需的实际梯度更新次数”。

单次 run：
```bash
python -m mc_dqn.sample_efficiency --run_dir runs/baseline_seed0 --run_dir runs/ours_seed0 --window 100 --threshold -150
```

多 seed 聚合（按 `baseline_seed*` / `ours_seed*` 自动分组并统计均值/方差）：
```bash
python -m mc_dqn.sample_efficiency --aggregate_seeds \
	--run_dir runs/baseline_seed0 --run_dir runs/baseline_seed1 --run_dir runs/baseline_seed2 \
	--run_dir runs/ours_seed0 --run_dir runs/ours_seed1 --run_dir runs/ours_seed2 \
	--window 100 --threshold -150
```

默认使用严格的 `>`（与论文表述一致）。如果你希望用 `>=`，加 `--inclusive`。

## 可视化渲染（山 + 旗帜）

先确保某个 run 已训练并生成 `q.pt`（训练脚本会自动保存）。

打开交互窗口（macOS 会弹出 pygame 窗口）：
```bash
python -m mc_dqn.render_policy --run_dir runs/ours_seed0 --episodes 3 --render human
```

录制 mp4（保存到 `runs/ours_seed0/videos/`）：
```bash
python -m mc_dqn.render_policy --run_dir runs/ours_seed0 --episodes 3 --record_video
```
