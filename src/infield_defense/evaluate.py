from __future__ import annotations

import csv
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
                "failure",
                "is_null",
                "outcome",
                "out_recorded",
                "runner_scored",
                "dead_ball_reason",
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


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    # Group by (method, n_robots, vmax, base_scale, tau_handoff)
    keys: dict[tuple[str, int, float, float, float], list[dict[str, str]]] = {}
    for r in rows:
        key = (
            r["method"],
            int(r["n_robots"]),
            float(r["vmax"]),
            float(r["base_scale"]),
            float(r["tau_handoff"]),
        )
        keys.setdefault(key, []).append(r)

    summary_rows: list[dict[str, float | int | str]] = []
    for (method, n, vmax, base_scale, tau), group in sorted(keys.items()):
        trial_count = len(group)
        success_count = sum(1 for g in group if g["outcome"] == "out")
        failure_count = sum(1 for g in group if g["outcome"] == "run")
        foul_ball_count = sum(1 for g in group if g["outcome"] == "foul")
        other_outcome_count = trial_count - success_count - failure_count - foul_ball_count
        non_foul_trials = trial_count - foul_ball_count
        resolved_group = [g for g in group if g["outcome"] in {"out", "run"}]
        totals = [float(g["t_total"]) for g in resolved_group]
        relays = [1.0 if g["relay_used"].lower() == "true" else 0.0 for g in resolved_group]
        success_pct = 100.0 * success_count / max(1, non_foul_trials)
        failure_pct = 100.0 * failure_count / max(1, non_foul_trials)
        foul_ball_pct = 100.0 * foul_ball_count / max(1, trial_count)
        relay_used_pct = 100.0 * float(np.mean(relays)) if relays else float("nan")
        mean, std, median, p90 = _summary_stats(totals)
        summary_rows.append(
            {
                "method": method,
                "n": n,
                "vmax": vmax,
                "base_scale": base_scale,
                "tau_handoff": tau,
                "trials": trial_count,
                "non_foul_trials": non_foul_trials,
                "resolved_trials": len(resolved_group),
                "successes": success_count,
                "failures": failure_count,
                "foul_balls": foul_ball_count,
                "other_outcomes": other_outcome_count,
                "success_pct": success_pct,
                "failure_pct": failure_pct,
                "foul_ball_pct": foul_ball_pct,
                "relay_used_pct": relay_used_pct,
                "t_total_mean": mean,
                "t_total_std": std,
                "t_total_med": median,
                "t_total_p90": p90,
            }
        )
    return summary_rows


def write_summary_csv(rows: list[dict[str, float | int | str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "n",
        "vmax",
        "base_scale",
        "tau_handoff",
        "trials",
        "non_foul_trials",
        "resolved_trials",
        "successes",
        "failures",
        "foul_balls",
        "other_outcomes",
        "success_pct",
        "failure_pct",
        "foul_ball_pct",
        "relay_used_pct",
        "t_total_mean",
        "t_total_std",
        "t_total_med",
        "t_total_p90",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv_from_trials_csv(trials_csv: Path, out_csv: Path) -> None:
    write_summary_csv(summarize_rows(_load_rows(trials_csv)), out_csv)


def print_table_from_csv(csv_path: Path) -> None:
    summary_rows = summarize_rows(_load_rows(csv_path))
    header = (
        "method,n,vmax,base_scale,tau_handoff,trials,non_foul_trials,resolved_trials,successes,"
        "failures,foul_balls,other_outcomes,success_pct,failure_pct,foul_ball_pct,relay_used_pct,"
        "t_total_mean,t_total_std,t_total_med,t_total_p90"
    )
    print(header)
    for row in summary_rows:
        print(
            f"{row['method']},{row['n']},{row['vmax']},{row['base_scale']},{row['tau_handoff']},"
            f"{row['trials']},{row['non_foul_trials']},{row['resolved_trials']},{row['successes']},"
            f"{row['failures']},{row['foul_balls']},{row['other_outcomes']},{row['success_pct']:.1f},"
            f"{row['failure_pct']:.1f},{row['foul_ball_pct']:.1f},{row['relay_used_pct']:.1f},"
            f"{row['t_total_mean']:.3f},{row['t_total_std']:.3f},{row['t_total_med']:.3f},"
            f"{row['t_total_p90']:.3f}"
        )


def main() -> None:
    """
    Default run:
      - writes `results/eval.csv`
      - writes `results/eval_summary.csv`
      - prints the aggregate table to stdout
    """
    trials_out = Path("results/eval.csv")
    summary_out = Path("results/eval_summary.csv")
    run_sweep(out_csv=trials_out, k_trials=50, seed=0)
    write_summary_csv_from_trials_csv(trials_out, summary_out)
    print_table_from_csv(trials_out)


if __name__ == "__main__":
    main()
