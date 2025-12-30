from __future__ import annotations

import argparse
from pathlib import Path

from .config import DensifyConfig, ExperimentConfig
from .train import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    p.add_argument("--out_dir", type=str, default="runs_2000")
    p.add_argument(
        "--variants",
        type=str,
        default="baseline,ours",
        help="Comma-separated variants: baseline,ours",
    )
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-150.0)
    p.add_argument(
        "--inclusive",
        action="store_true",
        help="Use >= threshold for sample-efficiency table/plots.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    densify_cfg = DensifyConfig()

    run_dirs: list[Path] = []
    for v in variants:
        for seed in seeds:
            cfg = ExperimentConfig(episodes=int(args.episodes), seed=seed)
            rd = train(v, cfg, densify_cfg, out_dir=str(out_dir))
            run_dirs.append(rd)

    # Generate plots + tables
    # Reward curve (mean±std) and sample-efficiency plot
    from .plot_results import main as plot_main
    from .sample_efficiency import main as eff_main
    import sys

    # plot_results
    argv = [
        "plot_results",
        "--aggregate_seeds",
        "--window",
        str(args.window),
        "--threshold",
        str(args.threshold),
        "--out_dir",
        str(out_dir),
    ]
    for rd in run_dirs:
        argv += ["--run_dir", str(rd)]
    sys.argv = argv
    plot_main()

    # sample_efficiency table
    out_csv = out_dir / "sample_efficiency_table.csv"
    argv = [
        "sample_efficiency",
        "--aggregate_seeds",
        "--window",
        str(args.window),
        "--threshold",
        str(args.threshold),
        "--out",
        str(out_csv),
    ]
    if args.inclusive:
        argv.append("--inclusive")
    for rd in run_dirs:
        argv += ["--run_dir", str(rd)]
    sys.argv = argv
    eff_main()

    print(f"All done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
