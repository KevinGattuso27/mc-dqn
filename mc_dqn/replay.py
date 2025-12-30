from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Tuple
import random
from collections import deque

import numpy as np


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool
    # n-step transition length (1 for standard DQN)
    n_steps: int = 1


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, k=batch_size)


class EpisodeBuffer:
    """Stores a single episode's raw (step-by-step) transitions."""

    def __init__(self) -> None:
        self._episode: List[Transition] = []

    def reset(self) -> None:
        self._episode = []

    def __len__(self) -> int:
        return len(self._episode)

    def push(self, t: Transition) -> None:
        self._episode.append(t)

    def as_list(self) -> List[Transition]:
        return list(self._episode)
