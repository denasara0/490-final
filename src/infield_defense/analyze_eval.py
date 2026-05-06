from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from infield_defense.evaluate import summarize_rows, write_summary_csv


def _load_trial_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def summarize_by_method(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary: list[dict[str, float | int | str]] = []
    for method, group in sorted(grouped.items()):
        total = len(group)
        outcomes = [row["outcome"] for row in group]
        foul_ball_count = sum(outcome == "foul" for outcome in outcomes)
        non_foul_trials = total - foul_ball_count
        resolved = [row for row in group if row["outcome"] in {"out", "run"}]
        t_totals = [_safe_float(row["t_total"]) for row in resolved]
        finite_totals = [value for value in t_totals if np.isfinite(value)]
        summary.append(
            {
                "method": method,
                "trials": total,
                "non_foul_trials": non_foul_trials,
                "successes": sum(outcome == "out" for outcome in outcomes),
                "failures": sum(outcome == "run" for outcome in outcomes),
                "foul_balls": foul_ball_count,
                "unresolved": sum(outcome == "unresolved" for outcome in outcomes),
                "success_pct": 100.0
                * sum(outcome == "out" for outcome in outcomes)
                / max(1, non_foul_trials),
                "failure_pct": 100.0
                * sum(outcome == "run" for outcome in outcomes)
                / max(1, non_foul_trials),
                "foul_ball_pct": 100.0 * foul_ball_count / max(1, total),
                "unresolved_pct": 100.0
                * sum(outcome == "unresolved" for outcome in outcomes)
                / max(1, total),
                "resolved_t_total_mean": (
                    float(np.mean(finite_totals)) if finite_totals else float("nan")
                ),
            }
        )
    return summary


def summarize_success_by_field(
    rows: list[dict[str, str]],
    field: str,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row[field])].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (method, field_value), group in sorted(grouped.items()):
        total = len(group)
        foul_ball_count = sum(row["outcome"] == "foul" for row in group)
        non_foul_trials = total - foul_ball_count
        summary.append(
            {
                "method": method,
                field: field_value,
                "trials": total,
                "non_foul_trials": non_foul_trials,
                "foul_balls": foul_ball_count,
                "success_pct": 100.0
                * sum(row["outcome"] == "out" for row in group)
                / max(1, non_foul_trials),
                "failure_pct": 100.0
                * sum(row["outcome"] == "run" for row in group)
                / max(1, non_foul_trials),
                "foul_ball_pct": 100.0 * foul_ball_count / max(1, total),
                "unresolved_pct": 100.0
                * sum(row["outcome"] == "unresolved" for row in group)
                / max(1, total),
            }
        )
    return summary


def _write_csv(rows: list[dict[str, float | int | str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_csv.open("w", newline="") as f:
            f.write("")
        return

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outcome_rates_by_method(
    method_summary_rows: list[dict[str, float | int | str]],
    out_path: Path,
) -> None:
    methods = [str(row["method"]) for row in method_summary_rows]
    success = [float(row["success_pct"]) for row in method_summary_rows]
    failure = [float(row["failure_pct"]) for row in method_summary_rows]
    foul_balls = [float(row["foul_ball_pct"]) for row in method_summary_rows]
    unresolved = [float(row["unresolved_pct"]) for row in method_summary_rows]

    x = np.arange(len(methods), dtype=np.float64)
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, success, width, label="Out %")
    ax.bar(x - 0.5 * width, failure, width, label="Run %")
    ax.bar(x + 0.5 * width, foul_balls, width, label="Foul %")
    ax.bar(x + 1.5 * width, unresolved, width, label="Unresolved %")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Percent of trials")
    ax.set_title("Success/Failure Rates and Foul Share by Method")
    ax.set_ylim(0.0, 100.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_success_curve(
    rows: list[dict[str, float | int | str]],
    *,
    x_field: str,
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    by_method: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append((float(row[x_field]), float(row["success_pct"])))

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, pairs in sorted(by_method.items()):
        pairs = sorted(pairs)
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=method)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.set_ylim(0.0, 100.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_resolved_total_time_boxplot(
    trial_rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in trial_rows:
        if row["outcome"] not in {"out", "run"}:
            continue
        t_total = _safe_float(row["t_total"])
        if np.isfinite(t_total):
            grouped[row["method"]].append(t_total)

    labels = sorted(grouped)
    values = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    if values:
        ax.boxplot(values, labels=labels)
    ax.set_ylabel("Resolved total time (s)")
    ax.set_title("Resolved Play Completion Time by Method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_eval_csv(
    trials_csv: Path,
    *,
    summary_csv: Path,
    analysis_dir: Path,
) -> list[Path]:
    rows = _load_trial_rows(trials_csv)
    summary_rows = summarize_rows(rows)
    method_summary_rows = summarize_by_method(rows)
    success_by_vmax_rows = summarize_success_by_field(rows, "vmax")
    success_by_n_rows = summarize_success_by_field(rows, "n_robots")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary_rows, summary_csv)

    method_summary_csv = analysis_dir / "method_summary.csv"
    success_by_vmax_csv = analysis_dir / "success_by_vmax.csv"
    success_by_n_csv = analysis_dir / "success_by_n_robots.csv"
    _write_csv(method_summary_rows, method_summary_csv)
    _write_csv(success_by_vmax_rows, success_by_vmax_csv)
    _write_csv(success_by_n_rows, success_by_n_csv)

    outcome_plot = analysis_dir / "outcome_rates_by_method.png"
    success_vmax_plot = analysis_dir / "success_vs_vmax.png"
    success_n_plot = analysis_dir / "success_vs_n_robots.png"
    t_total_plot = analysis_dir / "resolved_t_total_boxplot.png"

    plot_outcome_rates_by_method(method_summary_rows, outcome_plot)
    plot_success_curve(
        success_by_vmax_rows,
        x_field="vmax",
        x_label="vmax",
        title="Success Rate vs vmax",
        out_path=success_vmax_plot,
    )
    plot_success_curve(
        success_by_n_rows,
        x_field="n_robots",
        x_label="Number of robots",
        title="Success Rate vs Number of Robots",
        out_path=success_n_plot,
    )
    plot_resolved_total_time_boxplot(rows, t_total_plot)

    return [
        summary_csv,
        method_summary_csv,
        success_by_vmax_csv,
        success_by_n_csv,
        outcome_plot,
        success_vmax_plot,
        success_n_plot,
        t_total_plot,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze repeated-trial evaluation CSV output.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/eval.csv"),
        help="Path to the per-trial evaluation CSV.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/eval_summary.csv"),
        help="Path to write the aggregate summary CSV.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory for derived CSVs and plots.",
    )
    args = parser.parse_args()

    created_paths = analyze_eval_csv(
        args.input,
        summary_csv=args.summary_out,
        analysis_dir=args.analysis_dir,
    )

    print("Wrote analysis artifacts:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
