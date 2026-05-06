from __future__ import annotations

import csv
from pathlib import Path

from infield_defense.evaluate import print_table_from_csv, write_summary_csv_from_trials_csv


def test_print_table_tracks_success_failure_and_null_counts(
    tmp_path: Path,
    capsys,
) -> None:
    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", newline="") as f:
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
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 14.0,
                    "ball_v0_y": 14.0,
                    "success": True,
                    "failure": False,
                    "is_null": False,
                    "outcome": "out",
                    "out_recorded": True,
                    "runner_scored": False,
                    "dead_ball_reason": "",
                    "t_intercept": 1.0,
                    "t_delivery": 2.0,
                    "t_total": 3.0,
                    "relay_used": False,
                    "primary_idx": 1,
                    "relay_partner": -1,
                },
                {
                    "method": "ours_direct",
                    "trial": 1,
                    "n_robots": 4,
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 18.0,
                    "ball_v0_y": 0.0,
                    "success": False,
                    "failure": True,
                    "is_null": False,
                    "outcome": "run",
                    "out_recorded": False,
                    "runner_scored": True,
                    "dead_ball_reason": "",
                    "t_intercept": "nan",
                    "t_delivery": "nan",
                    "t_total": 5.0,
                    "relay_used": True,
                    "primary_idx": 0,
                    "relay_partner": 1,
                },
                {
                    "method": "ours_direct",
                    "trial": 2,
                    "n_robots": 4,
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 20.0,
                    "ball_v0_y": 0.0,
                    "success": False,
                    "failure": False,
                    "is_null": True,
                    "outcome": "foul",
                    "out_recorded": False,
                    "runner_scored": False,
                    "dead_ball_reason": "foul",
                    "t_intercept": "nan",
                    "t_delivery": "nan",
                    "t_total": 0.1,
                    "relay_used": False,
                    "primary_idx": 0,
                    "relay_partner": -1,
                },
            ]
        )

    print_table_from_csv(csv_path)

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("method,n,vmax,base_scale,tau_handoff,trials,non_foul_trials")

    row = lines[1].split(",")
    assert row[:5] == ["ours_direct", "4", "6.0", "1.0", "0.0"]
    assert row[5:12] == ["3", "2", "2", "1", "1", "1", "0"]
    assert row[12:16] == ["50.0", "50.0", "33.3", "50.0"]
    assert row[16:] == ["4.000", "1.000", "4.000", "4.800"]


def test_write_summary_csv_outputs_aggregate_csv(tmp_path: Path) -> None:
    trials_csv = tmp_path / "eval.csv"
    summary_csv = tmp_path / "eval_summary.csv"

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
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 14.0,
                    "ball_v0_y": 14.0,
                    "success": True,
                    "failure": False,
                    "is_null": False,
                    "outcome": "out",
                    "out_recorded": True,
                    "runner_scored": False,
                    "dead_ball_reason": "",
                    "t_intercept": 1.0,
                    "t_delivery": 2.0,
                    "t_total": 3.0,
                    "relay_used": False,
                    "primary_idx": 1,
                    "relay_partner": -1,
                },
                {
                    "method": "ours_direct",
                    "trial": 1,
                    "n_robots": 4,
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 18.0,
                    "ball_v0_y": 0.0,
                    "success": False,
                    "failure": True,
                    "is_null": False,
                    "outcome": "run",
                    "out_recorded": False,
                    "runner_scored": True,
                    "dead_ball_reason": "",
                    "t_intercept": "nan",
                    "t_delivery": "nan",
                    "t_total": 5.0,
                    "relay_used": True,
                    "primary_idx": 0,
                    "relay_partner": 1,
                },
                {
                    "method": "ours_direct",
                    "trial": 2,
                    "n_robots": 4,
                    "vmax": 6.0,
                    "base_scale": 1.0,
                    "base_x": 35.0,
                    "base_y": 0.0,
                    "tau_handoff": 0.0,
                    "ball_p0_x": 0.0,
                    "ball_p0_y": -35.0,
                    "ball_v0_x": 20.0,
                    "ball_v0_y": 0.0,
                    "success": False,
                    "failure": False,
                    "is_null": True,
                    "outcome": "foul",
                    "out_recorded": False,
                    "runner_scored": False,
                    "dead_ball_reason": "foul",
                    "t_intercept": "nan",
                    "t_delivery": "nan",
                    "t_total": 0.1,
                    "relay_used": False,
                    "primary_idx": 0,
                    "relay_partner": -1,
                },
            ]
        )

    write_summary_csv_from_trials_csv(trials_csv, summary_csv)

    with summary_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "ours_direct"
    assert row["n"] == "4"
    assert row["trials"] == "3"
    assert row["non_foul_trials"] == "2"
    assert row["resolved_trials"] == "2"
    assert row["successes"] == "1"
    assert row["failures"] == "1"
    assert row["foul_balls"] == "1"
    assert row["other_outcomes"] == "0"
    assert row["success_pct"] == "50.0"
    assert row["failure_pct"] == "50.0"
