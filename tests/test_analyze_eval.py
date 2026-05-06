from __future__ import annotations

import csv
from pathlib import Path

from infield_defense.analyze_eval import analyze_eval_csv


def test_analyze_eval_writes_summary_tables_and_plots(tmp_path: Path) -> None:
    trials_csv = tmp_path / "eval.csv"
    summary_csv = tmp_path / "eval_summary.csv"
    analysis_dir = tmp_path / "analysis"

    with trials_csv.open("w", newline="") as f:
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
        writer.writerows(
            [
                {
                    "method": "ours_direct",
                    "trial": 0,
                    "n_robots": 4,
                    "vmax": 4.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": -30.0,
                    "ball_p0_y": -5.0,
                    "ball_v0_x": 14.0,
                    "ball_v0_y": 2.0,
                    "success": True,
                    "failure": False,
                    "is_null": False,
                    "outcome": "out",
                    "out_recorded": True,
                    "runner_scored": False,
                    "dead_ball_reason": "",
                    "t_intercept": 1.5,
                    "t_delivery": 2.0,
                    "t_total": 3.5,
                    "relay_used": False,
                    "primary_idx": 1,
                    "relay_partner": -1,
                },
                {
                    "method": "ours_relay",
                    "trial": 0,
                    "n_robots": 4,
                    "vmax": 4.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": -30.0,
                    "ball_p0_y": -5.0,
                    "ball_v0_x": 14.0,
                    "ball_v0_y": 2.0,
                    "success": False,
                    "failure": True,
                    "is_null": False,
                    "outcome": "run",
                    "out_recorded": False,
                    "runner_scored": True,
                    "dead_ball_reason": "",
                    "t_intercept": 2.0,
                    "t_delivery": "nan",
                    "t_total": 5.5,
                    "relay_used": True,
                    "primary_idx": 0,
                    "relay_partner": 1,
                },
                {
                    "method": "nearest_direct",
                    "trial": 0,
                    "n_robots": 6,
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.25,
                    "ball_p0_x": -30.0,
                    "ball_p0_y": -5.0,
                    "ball_v0_x": 12.0,
                    "ball_v0_y": -1.0,
                    "success": False,
                    "failure": False,
                    "is_null": True,
                    "outcome": "foul",
                    "out_recorded": False,
                    "runner_scored": False,
                    "dead_ball_reason": "foul",
                    "t_intercept": "nan",
                    "t_delivery": "nan",
                    "t_total": 0.6,
                    "relay_used": False,
                    "primary_idx": 3,
                    "relay_partner": -1,
                },
            ]
        )

    created_paths = analyze_eval_csv(trials_csv, summary_csv=summary_csv, analysis_dir=analysis_dir)

    for path in created_paths:
        assert path.exists()

    with summary_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    nearest_row = next(row for row in rows if row["method"] == "nearest_direct")
    assert nearest_row["foul_balls"] == "1"
    assert nearest_row["non_foul_trials"] == "0"
    assert nearest_row["success_pct"] == "0.0"

    with (analysis_dir / "method_summary.csv").open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["method"] for row in rows} == {"nearest_direct", "ours_direct", "ours_relay"}
    ours_direct_row = next(row for row in rows if row["method"] == "ours_direct")
    assert ours_direct_row["success_pct"] == "100.0"
    assert ours_direct_row["foul_balls"] == "0"
