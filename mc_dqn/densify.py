from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import DensifyConfig
from .replay import Transition


def criticality(state: np.ndarray, cfg: DensifyConfig) -> float:
    """Energy-inspired criticality metric C(s).

    MountainCar observation is (position x, velocity v).

    A simple, stable choice:
    - potential_term: normalized height proxy from position
    - kinetic_term: v^2

    This is not the true environment potential energy, but it's monotonic w.r.t. x
    and works well for ranking "valuable" states in this task.
    """

    x = float(state[0])
    v = float(state[1])

    # normalize x from [-1.2, 0.6] to [0, 1]
    potential = (x + 1.2) / 1.8
    kinetic = v * v
    return cfg.w1 * potential + cfg.w2 * kinetic


def densify_episode(
    episode: List[Transition],
    gamma: float,
    cfg: DensifyConfig,
    episode_idx: int,
) -> List[Transition]:
    """Filter high-criticality states and reconnect them temporally.

    We keep top (100 - percentile)% states by criticality within the episode.
    Then we connect consecutive kept indices i -> j into a single transition with
    accumulated discounted reward.

    Action is taken from the first transition at i.
    Next-state is the state at j.
    Done is true if any step between i..j-1 terminated.
    """

    if len(episode) == 0:
        return []

    c_vals = np.array([criticality(t.s, cfg) for t in episode], dtype=np.float32)

    if episode_idx < cfg.warmup_episodes:
        # warm-up phase: keep all transitions to learn basic dynamics
        keep = np.ones(len(episode), dtype=bool)
    elif len(episode) < cfg.min_episode_len_for_threshold:
        keep = np.ones(len(episode), dtype=bool)
    else:
        delta = float(np.percentile(c_vals, cfg.percentile))
        keep = c_vals >= delta

    kept_indices = np.nonzero(keep)[0].tolist()
    if len(kept_indices) < 2:
        out: List[Transition] = []
        # Still include terminal transitions so sparse success signal is not lost.
        for t in episode:
            if bool(t.done):
                out.append(
                    Transition(
                        s=t.s,
                        a=int(t.a),
                        r=float(t.r),
                        s2=t.s2,
                        done=True,
                        n_steps=1,
                    )
                )
        return out

    out: List[Transition] = []

    for idx_pos in range(len(kept_indices) - 1):
        i = kept_indices[idx_pos]
        j = kept_indices[idx_pos + 1]

        n_steps = int(j - i)
        if n_steps <= 0:
            continue

        # accumulate discounted reward from i..j-1
        R = 0.0
        discount = 1.0
        done_any = False
        for k in range(i, j):
            R += discount * float(episode[k].r)
            discount *= gamma
            done_any = done_any or bool(episode[k].done)

        t_i = episode[i]
        t_j = episode[j]
        out.append(
            Transition(
                s=t_i.s,
                a=int(t_i.a),
                r=float(R),
                s2=t_j.s,
                done=bool(done_any) or bool(t_j.done),
                n_steps=n_steps,
            )
        )

    # Critical: always keep original terminal transitions (reward=0 on success) so sparse
    # success signal can propagate. This also ensures truncated episodes are treated as terminal.
    for t in episode:
        if bool(t.done):
            out.append(
                Transition(
                    s=t.s,
                    a=int(t.a),
                    r=float(t.r),
                    s2=t.s2,
                    done=True,
                    n_steps=1,
                )
            )

    return out
