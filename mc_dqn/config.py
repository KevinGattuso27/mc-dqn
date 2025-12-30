from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    env_id: str = "MountainCar-v0"
    seed: int = 0
    episodes: int = 800
    max_steps_per_episode: int = 200

    gamma: float = 0.99
    lr: float = 1e-3

    hidden_size: int = 64

    buffer_capacity: int = 50_000
    batch_size: int = 64

    # DQN stabilization
    target_update_interval: int = 500  # gradient steps

    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_episodes: int = 800  # linear decay across episodes

    # training schedule
    warmup_transitions: int = 1_000
    updates_per_episode: int | None = None  # if None, equals episode length


@dataclass(frozen=True)
class DensifyConfig:
    # C(s) = w1 * potential_term + w2 * kinetic_term
    w1: float = 1.0
    w2: float = 10.0

    # delta = 75th percentile
    percentile: float = 75.0

    # Warm-up: disable pruning for the first K episodes to avoid cold-start failure
    warmup_episodes: int = 200

    # Hybrid strategy to avoid distribution shift / forgetting:
    # - keep some raw 1-step transitions even after warm-up
    # - sample a mixture of core-set and raw buffer during training
    raw_keep_prob: float = 0.1
    core_sample_prob: float = 0.7

    # minimal number of transitions in an episode before computing percentile
    min_episode_len_for_threshold: int = 10
