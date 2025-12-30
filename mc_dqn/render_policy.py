from __future__ import annotations

import argparse
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch

from .model import QNetwork


@torch.no_grad()
def select_action(q: QNetwork, state: np.ndarray, device: torch.device, epsilon: float, n_actions: int) -> int:
    if epsilon > 0.0 and random.random() < epsilon:
        return random.randrange(n_actions)
    x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return int(torch.argmax(q(x), dim=1).item())


@torch.no_grad()
def run_episode(env, q: QNetwork, device: torch.device, *, epsilon: float = 0.0) -> float:
    s, _ = env.reset()
    s = np.asarray(s, dtype=np.float32)
    ep_reward = 0.0

    n_actions = int(env.action_space.n)

    while True:
        a = select_action(q, s, device, epsilon, n_actions)
        s2, r, terminated, truncated, _ = env.step(a)
        ep_reward += float(r)
        s = np.asarray(s2, dtype=np.float32)
        if terminated or truncated:
            break

    return ep_reward


@torch.no_grad()
def run_episode_collect_frames(
    env, q: QNetwork, device: torch.device, *, epsilon: float = 0.0
) -> tuple[float, list[np.ndarray]]:
    """Run one episode and return (episode_reward, frames).

    Requires env created with render_mode="rgb_array".
    """

    s, _ = env.reset()
    s = np.asarray(s, dtype=np.float32)
    frames: list[np.ndarray] = []

    n_actions = int(env.action_space.n)

    frame0 = env.render()
    if frame0 is not None:
        frames.append(np.asarray(frame0))

    ep_reward = 0.0
    while True:
        a = select_action(q, s, device, epsilon, n_actions)
        s2, r, terminated, truncated, _ = env.step(a)
        ep_reward += float(r)
        s = np.asarray(s2, dtype=np.float32)

        fr = env.render()
        if fr is not None:
            frames.append(np.asarray(fr))

        if terminated or truncated:
            break

    return ep_reward, frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="e.g. runs/ours_seed0")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None, help="Optional env reset seed.")
    p.add_argument(
        "--render",
        choices=["human", "rgb_array"],
        default="human",
        help="human opens a window; rgb_array is for wrappers like RecordVideo.",
    )
    p.add_argument(
        "--record_video",
        action="store_true",
        help="Record mp4 to <run_dir>/videos (uses rgb_array render mode).",
    )
    p.add_argument(
        "--record_success",
        action="store_true",
        help="Record ONE successful episode to mp4 (reward > -200). Failed attempts are discarded.",
    )
    p.add_argument(
        "--max_attempts",
        type=int,
        default=50,
        help="Max episodes to try when using --record_success.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 path for --record_success (default: <run_dir>/success.mp4).",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Optional epsilon exploration during rendering/recording (helps find a success).",
    )
    p.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to load: best_q.pt or q.pt.",
    )

    args = p.parse_args()

    run_dir = Path(args.run_dir)

    ckpt_best = run_dir / "best_q.pt"
    ckpt_last = run_dir / "q.pt"
    ckpt = ckpt_best if args.checkpoint == "best" else ckpt_last
    if not ckpt.exists():
        # fallback
        ckpt = ckpt_last if ckpt is ckpt_best else ckpt_best
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found in {run_dir}. Expected q.pt and/or best_q.pt. Please train first."
        )

    device = torch.device("cpu")

    # Create env
    if args.record_success:
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
    else:
        render_mode = "rgb_array" if args.record_video else args.render
        env = gym.make("MountainCar-v0", render_mode=render_mode)

        if args.record_video:
            video_dir = run_dir / "videos"
            env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), episode_trigger=lambda i: True)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    q = QNetwork(obs_dim, n_actions)
    q.load_state_dict(torch.load(ckpt, map_location="cpu"))
    q.eval().to(device)

    # Optional: seed reset
    if args.seed is not None:
        env.reset(seed=int(args.seed))

    if args.record_success:
        from moviepy.editor import ImageSequenceClip

        out_path = Path(args.out) if args.out else (run_dir / "success.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        success_reward = None
        success_frames: list[np.ndarray] | None = None

        for attempt in range(args.max_attempts):
            r, frames = run_episode_collect_frames(env, q, device, epsilon=float(args.epsilon))
            if r > -200.0:
                success_reward = r
                success_frames = frames
                break

        env.close()

        if success_frames is None:
            raise RuntimeError(
                f"No successful episode found within {args.max_attempts} attempts. "
                "Try increasing --max_attempts, or train a stronger checkpoint."
            )

        clip = ImageSequenceClip(success_frames, fps=int(args.fps))
        clip.write_videofile(str(out_path), codec="libx264", audio=False, verbose=False, logger=None)
        print(f"Saved success video: {out_path} (reward={success_reward})")
        return

    rewards = []
    for _ in range(args.episodes):
        rewards.append(run_episode(env, q, device, epsilon=float(args.epsilon)))

    env.close()
    print(f"Rendered {args.episodes} episode(s). Rewards: {rewards}")


if __name__ == "__main__":
    main()
