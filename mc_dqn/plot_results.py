from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def meillard_palette() -> dict[str, str]:
    """Warm 'Maillard' palette (caramel/coffee/cocoa).

    Keep colors muted and print-friendly.
    """

    return {
        "baseline": "#8D6E63",  # coffee brown
        "ours": "#C37B4B",  # caramel
        "default": "#6D4C41",  # dark cocoa
    }


def configure_matplotlib_style() -> None:
    # Minimal, clean style without requiring extra dependencies.
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def smooth(series: pd.Series, window: int = 100) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


@dataclass(frozen=True)
class RunSummary:
    name: str
    reached: bool
    episode_at_threshold: Optional[int]
    grad_updates_at_threshold: Optional[int]


def group_name_from_dir(run_dir: Path) -> str:
    """Infer variant/group name from run directory name.

    Expected naming: <variant>_seed<seed>, e.g. baseline_seed0, ours_seed2.
    If not matched, fallback to full directory name.
    """

    name = run_dir.name
    if "_seed" in name:
        return name.split("_seed", 1)[0]
    return name


def first_crossing(
    df: pd.DataFrame,
    window: int,
    threshold: float,
    *,
    inclusive: bool = True,
    censor_unreached: bool = True,
) -> tuple[bool, bool, Optional[int], Optional[int]]:
    """Return (reached, censored, episode_idx, grad_updates_total).

    reached: whether rolling-mean reward crosses threshold.
    censored: unreached but we return end-of-run values for plotting.

    By default we use >= (inclusive) to avoid edge cases where reward equals threshold.
    """

    if "episode_reward" not in df.columns:
        return False, False, None, None
    if "grad_updates_total" not in df.columns:
        return False, False, None, None

    r_smooth = smooth(df["episode_reward"], window)
    if inclusive:
        mask = (r_smooth >= threshold).to_numpy(dtype=bool)
    else:
        mask = (r_smooth > threshold).to_numpy(dtype=bool)
    hit = np.nonzero(mask)[0]
    if len(hit) == 0:
        if not censor_unreached:
            return False, False, None, None
        # Right-censor at end: show how many episodes/updates were spent without reaching.
        i_last = int(len(df) - 1)
        ep_last = int(df.loc[i_last, "episode"]) if "episode" in df.columns else i_last
        gu_last = int(df.loc[i_last, "grad_updates_total"])
        return False, True, ep_last, gu_last
    i = int(hit[0])
    ep = int(df.loc[i, "episode"]) if "episode" in df.columns else i
    gu = int(df.loc[i, "grad_updates_total"])
    return True, False, ep, gu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", action="append", required=True)
    p.add_argument("--window", type=int, default=100)
    p.add_argument(
        "--threshold",
        type=float,
        default=-150.0,
        help="Performance threshold on rolling-mean episode reward.",
    )
    p.add_argument(
        "--aggregate_seeds",
        action="store_true",
        help="Group run_dir by variant (prefix before _seed) and plot mean±std shading.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Use strict '>' instead of '>=' for threshold crossing.",
    )
    p.add_argument(
        "--no_censor",
        action="store_true",
        help="If not reached, leave sample-efficiency as missing (can look blank if all missing).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="runs",
        help="Directory to write figures and summary.csv",
    )
    args = p.parse_args()

    configure_matplotlib_style()
    palette = meillard_palette()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inclusive = not args.strict
    censor_unreached = not args.no_censor

    # Load all runs
    run_dfs: list[tuple[Path, pd.DataFrame]] = []
    for rd in args.run_dir:
        run_dir = Path(rd)
        df = pd.read_csv(run_dir / "metrics.csv")
        run_dfs.append((run_dir, df))

    summaries: list[RunSummary] = []

    # Figure 1: reward curves (optionally mean±std across seeds)
    plt.figure(figsize=(8.2, 4.6))

    if not args.aggregate_seeds:
        for run_dir, df in run_dfs:
            label = run_dir.name
            plt.plot(
                df["episode"],
                smooth(df["episode_reward"], args.window),
                label=label,
                linewidth=1.6,
            )

            reached, censored, ep_hit, gu_hit = first_crossing(
                df,
                args.window,
                args.threshold,
                inclusive=inclusive,
                censor_unreached=censor_unreached,
            )
            summaries.append(
                RunSummary(
                    name=label,
                    reached=reached,
                    episode_at_threshold=ep_hit,
                    grad_updates_at_threshold=gu_hit,
                )
            )
    else:
        # group by variant name
        groups: dict[str, list[pd.DataFrame]] = {}
        group_dirs: dict[str, list[Path]] = {}
        for run_dir, df in run_dfs:
            g = group_name_from_dir(run_dir)
            groups.setdefault(g, []).append(df)
            group_dirs.setdefault(g, []).append(run_dir)

        for gname, dfs in groups.items():
            # compute smoothed reward series per seed, then mean/std
            series_list: list[pd.Series] = []
            for df in dfs:
                s = smooth(df["episode_reward"], args.window)
                s.index = df["episode"].astype(int)
                series_list.append(s)

            # align on common episodes (intersection)
            common_idx = set(series_list[0].index.tolist())
            for s in series_list[1:]:
                common_idx &= set(s.index.tolist())
            common_eps = np.array(sorted(common_idx), dtype=int)

            if len(common_eps) == 0:
                continue

            mat = np.stack([s.reindex(common_eps).to_numpy(dtype=float) for s in series_list], axis=0)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)

            color = palette.get(gname, palette["default"])
            plt.plot(common_eps, mean, label=f"{gname} (mean)", color=color, linewidth=2.2)
            if mat.shape[0] >= 2:
                plt.fill_between(common_eps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

            # sample efficiency summary per seed, then aggregate
            seed_hits: list[tuple[bool, bool, Optional[int], Optional[int]]] = []
            for df in dfs:
                seed_hits.append(
                    first_crossing(
                        df,
                        args.window,
                        args.threshold,
                        inclusive=inclusive,
                        censor_unreached=censor_unreached,
                    )
                )

            reached_any = any(h[0] for h in seed_hits)
            # For plotting/aggregation: include censored values if enabled.
            ep_vals = [h[2] for h in seed_hits if h[2] is not None]
            gu_vals = [h[3] for h in seed_hits if h[3] is not None]

            # Store group-level mean for summary.csv (std will be produced separately below)
            summaries.append(
                RunSummary(
                    name=gname,
                    reached=reached_any,
                    episode_at_threshold=int(np.mean(ep_vals)) if ep_vals else None,
                    grad_updates_at_threshold=int(np.mean(gu_vals)) if gu_vals else None,
                )
            )

    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (window={args.window})")
    plt.title("MountainCar-v0: Average Reward over Episodes")
    plt.legend(frameon=False)
    out_reward = out_dir / ("compare_reward_mean_std.png" if args.aggregate_seeds else "compare_reward.png")
    plt.tight_layout()
    plt.savefig(out_reward, dpi=200)
    print(f"Saved: {out_reward}")

    # Sample efficiency visualization
    # Sample efficiency visualization (+ error bars when aggregating)
    names = [s.name for s in summaries]
    eps_vals = [np.nan if not s.reached else float(s.episode_at_threshold) for s in summaries]
    gu_vals = [np.nan if not s.reached else float(s.grad_updates_at_threshold) for s in summaries]

    eps_err = None
    gu_err = None
    if args.aggregate_seeds:
        # recompute std across seeds per group for error bars
        group_to_hits: dict[str, list[tuple[bool, bool, Optional[int], Optional[int]]]] = {}
        for run_dir, df in run_dfs:
            g = group_name_from_dir(run_dir)
            group_to_hits.setdefault(g, []).append(
                first_crossing(
                    df,
                    args.window,
                    args.threshold,
                    inclusive=inclusive,
                    censor_unreached=censor_unreached,
                )
            )

        eps_err_vals: list[float] = []
        gu_err_vals: list[float] = []
        for g in names:
            hits = group_to_hits.get(g, [])
            ep_list = [h[2] for h in hits if h[2] is not None]
            gu_list = [h[3] for h in hits if h[3] is not None]
            eps_err_vals.append(float(np.std(ep_list)) if len(ep_list) >= 2 else 0.0)
            gu_err_vals.append(float(np.std(gu_list)) if len(gu_list) >= 2 else 0.0)
        eps_err = np.array(eps_err_vals, dtype=float)
        gu_err = np.array(gu_err_vals, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    x = np.arange(len(names))

    bar_width = 0.45 if len(names) <= 3 else 0.35
    bar_colors = [palette.get(n, palette["default"]) for n in names]

    err_kw = {
        "ecolor": "#6B5B53",
        "elinewidth": 1.1,
        "capsize": 3,
        "capthick": 1.1,
    }

    axes[0].bar(
        x,
        gu_vals,
        yerr=gu_err,
        width=bar_width,
        color=bar_colors,
        edgecolor="#5D4037",
        linewidth=0.8,
        error_kw=err_kw,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("Gradient updates to threshold")
    axes[0].set_title(f"First reach: avgR(window={args.window}) > {args.threshold}")

    axes[1].bar(
        x,
        eps_vals,
        yerr=eps_err,
        width=bar_width,
        color=bar_colors,
        edgecolor="#5D4037",
        linewidth=0.8,
        error_kw=err_kw,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("Episodes to threshold")
    axes[1].set_title("First crossing episode")

    fig.tight_layout()
    out_eff = out_dir / ("sample_efficiency_mean_std.png" if args.aggregate_seeds else "sample_efficiency.png")
    fig.savefig(out_eff, dpi=200)
    print(f"Saved: {out_eff}")

    # CSV summary
    summary_path = out_dir / "summary.csv"
    # Compute reach_rate when aggregating seeds
    reach_rate_map: dict[str, float] = {}
    if args.aggregate_seeds:
        group_to_hits2: dict[str, list[tuple[bool, bool, Optional[int], Optional[int]]]] = {}
        for run_dir, df in run_dfs:
            g = group_name_from_dir(run_dir)
            group_to_hits2.setdefault(g, []).append(
                first_crossing(
                    df,
                    args.window,
                    args.threshold,
                    inclusive=inclusive,
                    censor_unreached=censor_unreached,
                )
            )
        for g, hits in group_to_hits2.items():
            if len(hits) == 0:
                continue
            reach_rate_map[g] = float(np.mean([1.0 if h[0] else 0.0 for h in hits]))

    rows = []
    for s in summaries:
        rows.append(
            {
                "run": s.name,
                "reached": s.reached,
                "reach_rate": reach_rate_map.get(s.name, ""),
                "episode_at_threshold": s.episode_at_threshold,
                "grad_updates_at_threshold": s.grad_updates_at_threshold,
                "threshold": args.threshold,
                "window": args.window,
                "inclusive": inclusive,
                "censored_unreached": censor_unreached,
            }
        )

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
