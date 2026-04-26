"""
Headless evaluation runner for repeated-trial metrics and baselines.

This is meant to generate the quantitative tables referenced in the report:
run K trials per setting, compare simple baselines, and write a CSV for plots.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from infield_defense.simulation import EpisodeConfig, run_episode


def _summary_stats(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    # Avoid noisy RuntimeWarnings for all-NaN groups (e.g., 0% success regimes).
    if arr.size == 0 or bool(np.all(np.isnan(arr))):
        nan = float("nan")
        return nan, nan, nan, nan
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    median = float(np.nanmedian(arr))
    p90 = float(np.nanpercentile(arr, 90))
    return mean, std, median, p90


def run_sweep(
    *,
    out_csv: Path,
    k_trials: int = 50,
    ns: tuple[int, ...] = (4, 6, 8),
    vmaxs: tuple[float, ...] = (4.0, 6.0, 8.0),
    base_scales: tuple[float, ...] = (1.0, 2.0),
    seed: int = 0,
    tau_handoffs: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> None:
    rng = np.random.default_rng(seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    methods = ("nearest_direct", "ours_direct", "ours_relay")

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "trial",
                "n_robots",
                "vmax",
                "base_scale",
                "base_x",
                "base_y",
                "tau_handoff",
                "ball_p0_x",
                "ball_p0_y",
                "ball_v0_x",
                "ball_v0_y",
                "success",
                "out_recorded",
                "runner_scored",
                "t_intercept",
                "t_delivery",
                "t_total",
                "relay_used",
                "primary_idx",
                "relay_partner",
            ],
        )
        writer.writeheader()

        for n in ns:
            for vmax in vmaxs:
                for base_scale in base_scales:
                    base_x = float(35.0 * base_scale)
                    base_y = 0.0
                    for tau_handoff in tau_handoffs:
                        for trial in range(k_trials):
                            # Randomize ball launch a bit per trial (kept simple and bounded).
                            p0 = (-30.0, -5.0 + float(rng.uniform(-2.0, 2.0)))
                            v0 = (float(rng.uniform(10.0, 18.0)), float(rng.uniform(-1.0, 4.0)))

                            cfg = EpisodeConfig(
                                n_robots=n,
                                vmax=vmax,
                                tau_handoff=float(tau_handoff),
                                base_pos=(base_x, base_y),
                                ball_p0=p0,
                                ball_v0=v0,
                            )

                            for method in methods:
                                metrics = run_episode(cfg, method=method)
                                writer.writerow(
                                    {
                                        "method": method,
                                        "trial": trial,
                                        "n_robots": n,
                                        "vmax": vmax,
                                        "base_scale": float(base_scale),
                                        "base_x": base_x,
                                        "base_y": base_y,
                                        "tau_handoff": float(tau_handoff),
                                        "ball_p0_x": p0[0],
                                        "ball_p0_y": p0[1],
                                        "ball_v0_x": v0[0],
                                        "ball_v0_y": v0[1],
                                        **metrics,
                                    }
                                )


def print_table_from_csv(csv_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by (method, n_robots, vmax, base_scale, tau_handoff)
    keys = {}
    for r in rows:
        key = (
            r["method"],
            int(r["n_robots"]),
            float(r["vmax"]),
            float(r["base_scale"]),
            float(r["tau_handoff"]),
        )
        keys.setdefault(key, []).append(r)

    print("method,n,vmax,base_scale,tau_handoff,success_pct,relay_used_pct,t_total_mean,t_total_std,t_total_med,t_total_p90")
    for (method, n, vmax, base_scale, tau), group in sorted(keys.items()):
        totals = [float(g["t_total"]) for g in group]
        successes = [1.0 if g["success"].lower() == "true" else 0.0 for g in group]
        relays = [1.0 if g["relay_used"].lower() == "true" else 0.0 for g in group]
        success_pct = 100.0 * float(np.mean(successes))
        relay_used_pct = 100.0 * float(np.mean(relays))
        mean, std, median, p90 = _summary_stats(totals)
        print(
            f"{method},{n},{vmax},{base_scale},{tau},"
            f"{success_pct:.1f},{relay_used_pct:.1f},"
            f"{mean:.3f},{std:.3f},{median:.3f},{p90:.3f}"
        )


def main() -> None:
    """
    Default run:
      - writes `results/eval.csv`
      - prints an aggregate table to stdout
    """
    out = Path("results/eval.csv")
    run_sweep(out_csv=out, k_trials=50, seed=0)
    print_table_from_csv(out)


if __name__ == "__main__":
    main()
