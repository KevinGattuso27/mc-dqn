from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .config import DensifyConfig, ExperimentConfig
from .densify import densify_episode
from .model import QNetwork
from .replay import EpisodeBuffer, ReplayBuffer, Transition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def select_action(
    q: QNetwork, state: np.ndarray, eps: float, n_actions: int, device: torch.device
) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return int(torch.argmax(q(x), dim=1).item())


def compute_epsilon(cfg: ExperimentConfig, episode_idx: int) -> float:
    # linear decay across episodes
    t = min(1.0, episode_idx / max(1, cfg.eps_decay_episodes - 1))
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def dqn_update(
    q: QNetwork,
    q_tgt: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    gamma: float,
    device: torch.device,
) -> float:
    s = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=device)
    a = torch.tensor([t.a for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    r = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    s2 = torch.tensor(np.stack([t.s2 for t in batch]), dtype=torch.float32, device=device)
    done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    n_steps = torch.tensor([t.n_steps for t in batch], dtype=torch.int64, device=device).unsqueeze(1)

    q_sa = q(s).gather(1, a)

    with torch.no_grad():
        max_q_next = torch.max(q_tgt(s2), dim=1, keepdim=True).values
        gamma_n = torch.pow(torch.tensor(gamma, dtype=torch.float32, device=device), n_steps.to(torch.float32))
        target = r + (1.0 - done) * gamma_n * max_q_next

    loss = nn.functional.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def write_config(run_dir: Path, cfg: ExperimentConfig, densify_cfg: DensifyConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write("ExperimentConfig\n")
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")
        f.write("\nDensifyConfig\n")
        for k, v in asdict(densify_cfg).items():
            f.write(f"{k}: {v}\n")


def train(
    variant: Literal["baseline", "ours"],
    cfg: ExperimentConfig,
    densify_cfg: DensifyConfig,
    *,
    out_dir: str = "runs",
) -> Path:
    set_seed(cfg.seed)

    run_dir = Path(out_dir) / f"{variant}_seed{cfg.seed}"
    write_config(run_dir, cfg, densify_cfg)

    env = gym.make(cfg.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    device = torch.device("cpu")

    q = QNetwork(obs_dim, n_actions, hidden_size=cfg.hidden_size).to(device)
    q_tgt = QNetwork(obs_dim, n_actions, hidden_size=cfg.hidden_size).to(device)
    q_tgt.load_state_dict(q.state_dict())

    optimizer = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    if variant == "baseline":
        buffer = ReplayBuffer(cfg.buffer_capacity)
    else:
        core = ReplayBuffer(cfg.buffer_capacity)
        raw = ReplayBuffer(cfg.buffer_capacity)
        episode_buf = EpisodeBuffer()

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_reward",
                "eps",
                "buffer_size",
                "grad_updates_total",
                "loss_mean",
            ],
        )
        writer.writeheader()

        grad_updates_total = 0
        target_update_counter = 0
        best_episode_reward = float("-inf")

        for ep in range(cfg.episodes):
            s, _ = env.reset(seed=cfg.seed + ep)
            s = np.asarray(s, dtype=np.float32)

            eps = compute_epsilon(cfg, ep)
            ep_reward = 0.0
            loss_vals: list[float] = []

            if variant == "ours":
                episode_buf.reset()

            for step in range(cfg.max_steps_per_episode):
                a = select_action(q, s, eps, n_actions, device)
                s2, r, terminated, truncated, _ = env.step(a)
                done = bool(terminated or truncated)
                s2 = np.asarray(s2, dtype=np.float32)

                ep_reward += float(r)

                t = Transition(s=s, a=int(a), r=float(r), s2=s2, done=done)

                if variant == "baseline":
                    buffer.push(t)
                else:
                    episode_buf.push(t)
                    # Hybrid replay: keep some raw 1-step transitions to preserve coverage
                    # of early/low-energy states (prevents later collapse back to -200).
                    if ep < densify_cfg.warmup_episodes or random.random() < densify_cfg.raw_keep_prob:
                        raw.push(t)

                s = s2

                if done:
                    break

            # Build core-set transitions once per episode, then train from core buffer.
            if variant == "ours":
                densified = densify_episode(
                    episode_buf.as_list(),
                    gamma=cfg.gamma,
                    cfg=densify_cfg,
                    episode_idx=ep,
                )
                for t2 in densified:
                    core.push(t2)

            # Training updates
            if variant == "baseline":
                train_buf = buffer
            else:
                train_buf = core

            updates = cfg.updates_per_episode
            if updates is None:
                updates = step + 1

            if variant == "ours":
                effective_size = max(len(core), len(raw))
            else:
                effective_size = len(train_buf)

            if effective_size >= max(cfg.warmup_transitions, cfg.batch_size):
                for _ in range(int(updates)):
                    if variant == "ours":
                        # Prefer core transitions but fall back to raw when core is too small
                        use_core = (len(core) >= cfg.batch_size) and (random.random() < densify_cfg.core_sample_prob)
                        source = core if use_core else raw
                        if len(source) < cfg.batch_size:
                            # if one buffer is too small, try the other
                            source = raw if source is core else core
                        if len(source) < cfg.batch_size:
                            break
                        batch = source.sample(cfg.batch_size)
                    else:
                        batch = train_buf.sample(cfg.batch_size)
                    loss = dqn_update(q, q_tgt, optimizer, batch, cfg.gamma, device)
                    loss_vals.append(loss)

                    grad_updates_total += 1
                    target_update_counter += 1
                    if target_update_counter >= cfg.target_update_interval:
                        q_tgt.load_state_dict(q.state_dict())
                        target_update_counter = 0

            writer.writerow(
                {
                    "episode": ep,
                    "episode_reward": ep_reward,
                    "eps": eps,
                    "buffer_size": len(train_buf) if variant == "baseline" else len(core),
                    "grad_updates_total": grad_updates_total,
                    "loss_mean": float(np.mean(loss_vals)) if loss_vals else "",
                }
            )
            f.flush()

            # Save best checkpoint by episode return (useful for later rendering)
            if ep_reward > best_episode_reward:
                best_episode_reward = ep_reward
                torch.save(q.state_dict(), run_dir / "best_q.pt")

    env.close()

    # Save model for later rendering/evaluation
    torch.save(q.state_dict(), run_dir / "q.pt")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "ours"], required=True)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs")

    args = p.parse_args()

    cfg = ExperimentConfig(episodes=args.episodes, seed=args.seed)
    densify_cfg = DensifyConfig()

    run_dir = train(args.variant, cfg, densify_cfg, out_dir=args.out_dir)
    print(f"Done. Logs in: {run_dir}")


if __name__ == "__main__":
    main()
