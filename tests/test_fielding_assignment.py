from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from infield_defense.cli import compute_live_fielding_decision
from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD
from infield_defense.coordination import plan_fielding_assignment
from infield_defense.simulation import (
    current_ball_pos,
    launch_ground_ball,
    make_state,
    plan_current_fielding_decision,
)


def _default_positions() -> np.ndarray:
    return np.array(list(DEFAULT_FIELD.infielder_positions().values()), dtype=np.float64)


def _first_base_line_ball_kwargs() -> dict[str, np.ndarray | float]:
    return {
        "t_now": 0.0,
        "ball_t0": 0.0,
        "ball_p0": np.array([0.0, -35.0], dtype=np.float64),
        "ball_v0": np.array([14.0, 14.0], dtype=np.float64),
        "ball_a": np.array([0.0, 0.0], dtype=np.float64),
        "vmax": DEFAULT_DYNAMICS.vmax,
        "sample_dt_s": DEFAULT_DYNAMICS.dt,
    }


def test_first_base_line_trajectory_prefers_first_baseman() -> None:
    home_positions = _default_positions()
    decision = plan_fielding_assignment(
        home_positions, home_positions, **_first_base_line_ball_kwargs()
    )

    assert decision.trajectory_owner_idx == 1
    assert decision.primary_idx == 1
    assert np.allclose(decision.primary_intercept_point, decision.intercept_points[1])


def test_continuous_replanning_keeps_first_baseman_on_play() -> None:
    state = make_state(n_robots=4, field=DEFAULT_FIELD)
    launch_ground_ball(
        state,
        0.0,
        p0=np.array([0.0, -35.0], dtype=np.float64),
        v0=np.array([14.0, 14.0], dtype=np.float64),
        decel_magnitude=0.0,
        vmax=DEFAULT_DYNAMICS.vmax,
    )

    for step_idx in range(1, 6):
        t_now = step_idx * DEFAULT_DYNAMICS.dt
        decision = plan_current_fielding_decision(
            state, t_now, vmax=DEFAULT_DYNAMICS.vmax, sample_dt_s=DEFAULT_DYNAMICS.dt
        )
        assert decision is not None
        assert decision.primary_idx == 1

        target = decision.primary_intercept_point
        delta = target - state.positions[decision.primary_idx]
        distance = float(np.linalg.norm(delta))
        if distance > 1e-9:
            direction = delta / distance
            step = min(distance, DEFAULT_DYNAMICS.vmax * DEFAULT_DYNAMICS.dt)
            state.positions[decision.primary_idx] += direction * step


def test_non_owner_can_override_when_owner_is_too_slow() -> None:
    home_positions = _default_positions()
    positions = home_positions.copy()
    positions[1] = np.array([60.0, -20.0], dtype=np.float64)

    decision = plan_fielding_assignment(positions, home_positions, **_first_base_line_ball_kwargs())

    assert decision.trajectory_owner_idx == 1
    assert decision.primary_idx == 0
    assert decision.intercept_times[0] + 0.50 < decision.intercept_times[1]


def test_cli_and_simulation_share_same_fielding_decision() -> None:
    state = make_state(n_robots=4, field=DEFAULT_FIELD)
    launch_ground_ball(
        state,
        0.0,
        p0=np.array([0.0, -35.0], dtype=np.float64),
        v0=np.array([14.0, 14.0], dtype=np.float64),
        decel_magnitude=0.0,
        vmax=DEFAULT_DYNAMICS.vmax,
    )

    sim_decision = plan_current_fielding_decision(
        state,
        0.25,
        vmax=DEFAULT_DYNAMICS.vmax,
        sample_dt_s=DEFAULT_DYNAMICS.dt,
    )
    live_decision = compute_live_fielding_decision(state, 0.25, dyn=DEFAULT_DYNAMICS)

    assert sim_decision is not None
    assert live_decision is not None
    assert live_decision.primary_idx == sim_decision.primary_idx
    assert live_decision.trajectory_owner_idx == sim_decision.trajectory_owner_idx
    assert np.allclose(live_decision.primary_intercept_point, sim_decision.primary_intercept_point)


def test_primary_targets_predicted_intercept_point_not_current_ball_position() -> None:
    state = make_state(n_robots=4, field=DEFAULT_FIELD)
    launch_ground_ball(
        state,
        0.0,
        p0=np.array([0.0, -35.0], dtype=np.float64),
        v0=np.array([14.0, 14.0], dtype=np.float64),
        decel_magnitude=0.0,
        vmax=DEFAULT_DYNAMICS.vmax,
    )

    decision = plan_current_fielding_decision(
        state,
        0.25,
        vmax=DEFAULT_DYNAMICS.vmax,
        sample_dt_s=DEFAULT_DYNAMICS.dt,
    )
    ball_now = current_ball_pos(state, 0.25)

    assert decision is not None
    assert not np.allclose(decision.primary_intercept_point, ball_now)
    assert decision.primary_intercept_point[0] > ball_now[0]
