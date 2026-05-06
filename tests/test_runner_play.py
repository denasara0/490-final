from __future__ import annotations

import math

import numpy as np

from infield_defense.baserunning import (
    advance_runner,
    current_target_base_name,
    make_runner,
    start_runner,
)
from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD
from infield_defense.simulation import (
    EpisodeConfig,
    make_play_state,
    make_state,
    run_episode,
    step_live_play,
)


def test_runner_reaches_first_after_expected_travel_time() -> None:
    runner = make_runner(field=DEFAULT_FIELD)
    start_runner(runner, field=DEFAULT_FIELD)

    home = np.array(DEFAULT_FIELD.base_positions()["home"], dtype=np.float64)
    first = np.array(DEFAULT_FIELD.base_positions()["first"], dtype=np.float64)
    travel_time = float(np.linalg.norm(first - home)) / DEFAULT_DYNAMICS.vmax

    reached = advance_runner(
        runner,
        dt=travel_time,
        speed=DEFAULT_DYNAMICS.vmax,
        field=DEFAULT_FIELD,
        touch_radius=0.0,
    )

    assert reached == "first"
    assert np.allclose(runner.position, first)
    assert current_target_base_name(runner, field=DEFAULT_FIELD) == "second"


def test_runner_switches_to_second_immediately_after_touching_first() -> None:
    runner = make_runner(field=DEFAULT_FIELD)
    start_runner(runner, field=DEFAULT_FIELD)

    home = np.array(DEFAULT_FIELD.base_positions()["home"], dtype=np.float64)
    first = np.array(DEFAULT_FIELD.base_positions()["first"], dtype=np.float64)
    travel_time = float(np.linalg.norm(first - home)) / DEFAULT_DYNAMICS.vmax
    advance_runner(
        runner,
        dt=travel_time,
        speed=DEFAULT_DYNAMICS.vmax,
        field=DEFAULT_FIELD,
        touch_radius=0.0,
    )

    prev_position = runner.position.copy()
    advance_runner(
        runner,
        dt=DEFAULT_DYNAMICS.dt,
        speed=DEFAULT_DYNAMICS.vmax,
        field=DEFAULT_FIELD,
        touch_radius=0.0,
    )

    assert current_target_base_name(runner, field=DEFAULT_FIELD) == "second"
    assert runner.position[1] > prev_position[1]


def test_force_target_shifts_to_second_when_runner_beats_force_at_first() -> None:
    state = make_state(field=DEFAULT_FIELD)
    play = make_play_state()
    start_runner(state.runner, field=DEFAULT_FIELD)
    play.has_ball = True
    play.holder_idx = 0
    state.positions[0] = np.array([-35.0, 0.0], dtype=np.float64)

    reached_second_target = False
    for step in range(220):
        t_now = (step + 1) * DEFAULT_DYNAMICS.dt
        step_live_play(
            state,
            play,
            t_now,
            method="ours_direct",
            dt=DEFAULT_DYNAMICS.dt,
            vmax=DEFAULT_DYNAMICS.vmax,
            intercept_radius=1.5,
            base_radius=0.75,
            handoff_radius=0.75,
            tau_handoff=0.0,
        )
        if current_target_base_name(state.runner, field=DEFAULT_FIELD) == "second":
            reached_second_target = True
            break

    assert reached_second_target
    assert not play.out_recorded
    assert not play.runner_scored


def test_run_episode_records_runner_aware_force_out() -> None:
    cfg = EpisodeConfig(
        n_robots=4,
        max_time=12.0,
        launch_time=0.0,
        ball_p0=(0.0, -35.0),
        ball_v0=(14.0, 14.0),
        ball_decel=0.0,
    )

    metrics = run_episode(cfg, method="ours_direct")

    assert metrics["success"] is True
    assert metrics["failure"] is False
    assert metrics["is_null"] is False
    assert metrics["outcome"] == "out"
    assert metrics["out_recorded"] is True
    assert metrics["runner_scored"] is False
    assert math.isfinite(float(metrics["t_intercept"]))
    assert math.isfinite(float(metrics["t_total"]))


def test_run_episode_runner_can_score_before_any_out() -> None:
    cfg = EpisodeConfig(
        n_robots=4,
        max_time=40.0,
        launch_time=0.0,
        ball_p0=(60.0, 60.0),
        ball_v0=(0.0, 0.0),
        ball_decel=0.0,
    )

    metrics = run_episode(cfg, method="ours_direct")

    assert metrics["success"] is False
    assert metrics["failure"] is True
    assert metrics["is_null"] is False
    assert metrics["outcome"] == "run"
    assert metrics["out_recorded"] is False
    assert metrics["runner_scored"] is True
    assert math.isfinite(float(metrics["t_total"]))


def test_run_episode_records_foul_ball_as_null_outcome() -> None:
    cfg = EpisodeConfig(
        n_robots=4,
        max_time=2.0,
        launch_time=0.0,
        ball_p0=DEFAULT_FIELD.home_base,
        ball_v0=(20.0, 0.0),
        ball_decel=0.0,
    )

    metrics = run_episode(cfg, method="ours_direct")

    assert metrics["success"] is False
    assert metrics["failure"] is False
    assert metrics["is_null"] is True
    assert metrics["outcome"] == "foul"
    assert metrics["out_recorded"] is False
    assert metrics["runner_scored"] is False
    assert metrics["dead_ball_reason"] == "foul"
    assert math.isnan(float(metrics["t_intercept"]))
    assert math.isfinite(float(metrics["t_total"]))


def test_relay_method_still_resolves_play_with_runner_model() -> None:
    cfg = EpisodeConfig(
        n_robots=4,
        max_time=20.0,
        launch_time=0.0,
        ball_p0=(-35.0, 0.0),
        ball_v0=(5.0, 0.0),
        ball_decel=0.0,
    )

    metrics = run_episode(cfg, method="ours_relay")

    assert metrics["out_recorded"] is True
    assert metrics["runner_scored"] is False
    assert metrics["relay_used"] is True
    assert metrics["relay_partner"] == 1


def test_relay_prefers_covering_second_baseman_when_second_is_forced() -> None:
    state = make_state(field=DEFAULT_FIELD)
    play = make_play_state()
    start_runner(state.runner, field=DEFAULT_FIELD)
    state.runner.position = np.array(DEFAULT_FIELD.base_positions()["first"], dtype=np.float64)
    state.runner.next_base_index = 2
    play.has_ball = True
    play.holder_idx = 0
    state.positions[0] = np.array([-30.0, 5.0], dtype=np.float64)

    step_live_play(
        state,
        play,
        DEFAULT_DYNAMICS.dt,
        method="ours_relay",
        dt=DEFAULT_DYNAMICS.dt,
        vmax=DEFAULT_DYNAMICS.vmax,
        intercept_radius=1.5,
        base_radius=0.75,
        handoff_radius=0.75,
        tau_handoff=0.0,
        pass_speed=DEFAULT_DYNAMICS.pass_speed,
    )

    assert play.selected_relay_partner == 2
    assert play.pass_in_flight is True
