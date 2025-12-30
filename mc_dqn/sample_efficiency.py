from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EfficiencyResult:
    run: str
    reached: bool
    episode_at_threshold: Optional[int]
    grad_updates_at_threshold: Optional[int]


def group_name_from_dir(run_dir: Path) -> str:
    name = run_dir.name
    if "_seed" in name:
        return name.split("_seed", 1)[0]
    return name


def first_reach_updates(
    df: pd.DataFrame,
    *,
    window: int,
    threshold: float,
    inclusive: bool,
) -> tuple[bool, Optional[int], Optional[int], Optional[int]]:
    """Find first episode where rolling-mean reward crosses threshold.

    Returns (reached, episode, grad_updates_total, env_steps_total).

    Note: uses the logged per-episode cumulative `grad_updates_total` as the
    'actual gradient updates consumed so far'.
    """

    if "episode_reward" not in df.columns or "grad_updates_total" not in df.columns:
        return False, None, None, None

    # Estimate per-episode environment steps from return.
    # MountainCar-v0 uses reward=-1 each step until goal, and 0 on success step.
    # So for success: return = -(T-1)  =>  T = -return + 1
    # For timeout/failure: return == -200 => T=200.
    def ep_steps_from_return(ret: float) -> int:
        if ret <= -200.0:
            return 200
        t = int(round(-float(ret))) + 1
        return max(1, min(200, t))

    steps_per_ep = df["episode_reward"].map(ep_steps_from_return).astype(int)
    env_steps_total = steps_per_ep.cumsum()

    r = df["episode_reward"].rolling(window=window, min_periods=1).mean()
    mask = (r >= threshold) if inclusive else (r > threshold)
    hit = np.nonzero(mask.to_numpy(dtype=bool))[0]
    if len(hit) == 0:
        return False, None, None, None

    i = int(hit[0])
    ep = int(df.loc[i, "episode"]) if "episode" in df.columns else i
    gu = int(df.loc[i, "grad_updates_total"])
    env_steps = int(env_steps_total.iloc[i])
    return True, ep, gu, env_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", action="append", required=True)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-150.0)
    p.add_argument(
        "--inclusive",
        action="store_true",
        help="Use >= threshold (default uses strict > to match论文表述).",
    )
    p.add_argument(
        "--aggregate_seeds",
        action="store_true",
        help="Group by prefix before _seed and report mean/std across seeds.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="runs/sample_efficiency_table.csv",
        help="Output CSV path.",
    )
    args = p.parse_args()

    run_dfs: list[tuple[Path, pd.DataFrame]] = []
    for rd in args.run_dir:
        run_dir = Path(rd)
        df = pd.read_csv(run_dir / "metrics.csv")
        run_dfs.append((run_dir, df))

    results: list[dict] = []

    if not args.aggregate_seeds:
        for run_dir, df in run_dfs:
            reached, ep, gu, env_steps = first_reach_updates(
                df,
                window=int(args.window),
                threshold=float(args.threshold),
                inclusive=bool(args.inclusive),
            )
            results.append(
                {
                    "run": run_dir.name,
                    "reached": reached,
                    "episode_at_threshold": ep,
                    "grad_updates_at_threshold": gu,
                    "env_steps_at_threshold": env_steps,
                    "threshold": float(args.threshold),
                    "window": int(args.window),
                    "inclusive": bool(args.inclusive),
                }
            )
    else:
        groups: dict[str, list[pd.DataFrame]] = {}
        for run_dir, df in run_dfs:
            groups.setdefault(group_name_from_dir(run_dir), []).append(df)

        for gname, dfs in sorted(groups.items(), key=lambda x: x[0]):
            per_seed: list[tuple[bool, Optional[int], Optional[int], Optional[int]]] = []
            for df in dfs:
                per_seed.append(
                    first_reach_updates(
                        df,
                        window=int(args.window),
                        threshold=float(args.threshold),
                        inclusive=bool(args.inclusive),
                    )
                )

            reached_flags = [r[0] for r in per_seed]
            gu_vals = [r[2] for r in per_seed if r[0] and r[2] is not None]
            ep_vals = [r[1] for r in per_seed if r[0] and r[1] is not None]
            env_vals = [r[3] for r in per_seed if r[0] and r[3] is not None]

            results.append(
                {
                    "run": gname,
                    "seeds": len(per_seed),
                    "reach_rate": float(np.mean([1.0 if f else 0.0 for f in reached_flags])),
                    "episode_at_threshold_mean": float(np.mean(ep_vals)) if ep_vals else "",
                    "episode_at_threshold_std": float(np.std(ep_vals)) if len(ep_vals) >= 2 else "",
                    "grad_updates_at_threshold_mean": float(np.mean(gu_vals)) if gu_vals else "",
                    "grad_updates_at_threshold_std": float(np.std(gu_vals)) if len(gu_vals) >= 2 else "",
                    "env_steps_at_threshold_mean": float(np.mean(env_vals)) if env_vals else "",
                    "env_steps_at_threshold_std": float(np.std(env_vals)) if len(env_vals) >= 2 else "",
                    "threshold": float(args.threshold),
                    "window": int(args.window),
                    "inclusive": bool(args.inclusive),
                }
            )

        # Add ratios vs baseline when available and both reached
        # (blank if baseline didn't reach)
        baseline = next((r for r in results if r.get("run") == "baseline"), None)
        if baseline is not None:
            b_gu = baseline.get("grad_updates_at_threshold_mean")
            b_env = baseline.get("env_steps_at_threshold_mean")
            for r in results:
                if r.get("run") == "baseline":
                    r["grad_updates_ratio_vs_baseline"] = 1.0 if b_gu not in ("", None) else ""
                    r["env_steps_ratio_vs_baseline"] = 1.0 if b_env not in ("", None) else ""
                    continue
                gu = r.get("grad_updates_at_threshold_mean")
                env = r.get("env_steps_at_threshold_mean")
                if b_gu in ("", None) or gu in ("", None):
                    r["grad_updates_ratio_vs_baseline"] = ""
                else:
                    r["grad_updates_ratio_vs_baseline"] = float(gu) / float(b_gu)
                if b_env in ("", None) or env in ("", None):
                    r["env_steps_ratio_vs_baseline"] = ""
                else:
                    r["env_steps_ratio_vs_baseline"] = float(env) / float(b_env)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)

    # Console summary (compact)
    print(f"Saved: {out_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
