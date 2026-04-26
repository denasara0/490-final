from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from infield_defense.ball import ball_position
from infield_defense.baserunning import (
    RunnerState,
    advance_runner,
    current_target_base_name,
    current_target_base_position,
    make_runner,
    mark_runner_out,
    start_runner,
)
from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD, DynamicsConfig, FieldConfig
from infield_defense.coordination import FieldingDecision, delivery_costs, plan_fielding_assignment
from infield_defense.formation import formation_control_input, neighbor_indices_ring


@dataclass
class BallState:
    p0: npt.NDArray[np.float64]
    v0: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    t0: float = 0.0
    active: bool = False


@dataclass
class SimState:
    positions: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    home_positions: npt.NDArray[np.float64]
    ball: BallState
    runner: RunnerState
    primary_idx: int | None = None
    role_names: list[str] = field(default_factory=list)
    neighbors: list[list[int]] = field(default_factory=list)


@dataclass
class PlayState:
    launched: bool = False
    has_ball: bool = False
    holder_idx: int | None = None
    relay_partner: int | None = None
    selected_relay_partner: int | None = None
    relay_used: bool = False
    relay_target_base: str | None = None
    pass_in_flight: bool = False
    pass_start_time: float | None = None
    pass_arrival_time: float | None = None
    pass_start_pos: npt.NDArray[np.float64] | None = None
    pass_end_pos: npt.NDArray[np.float64] | None = None
    handoff_done: bool = False
    handoff_timer: float = 0.0
    intercept_time: float | None = None
    delivery_time: float | None = None
    total_time: float | None = None
    out_recorded: bool = False
    runner_scored: bool = False
    dead_ball_reason: str | None = None
    result_text: str = "Waiting for pitch"

    @property
    def resolved(self) -> bool:
        return self.out_recorded or self.runner_scored or self.dead_ball_reason is not None


@dataclass(frozen=True)
class EpisodeConfig:
    """
    Parameters for a single headless simulation episode, for repeated-trial evaluation.
    """

    n_robots: int = 5
    vmax: float = DEFAULT_DYNAMICS.vmax
    formation_gain: float = DEFAULT_DYNAMICS.formation_gain
    dt: float = DEFAULT_DYNAMICS.dt
    launch_time: float = 0.5

    ball_p0: tuple[float, float] = (-30.0, -5.0)
    ball_v0: tuple[float, float] = (14.0, 2.0)
    ball_decel: float = DEFAULT_DYNAMICS.ball_decel

    intercept_radius: float = 0.5
    base_radius: float = 0.75
    handoff_radius: float = 0.75
    tau_handoff: float = 0.0
    pass_speed: float = DEFAULT_DYNAMICS.pass_speed
    base_pos: tuple[float, float] = DEFAULT_FIELD.first_base

    max_time: float = 20.0


def _clip_to_field(positions: npt.NDArray[np.float64], field: FieldConfig) -> None:
    positions[:, 0] = np.clip(positions[:, 0], field.x_min, field.x_max)
    positions[:, 1] = np.clip(positions[:, 1], field.y_min, field.y_max)


def _unit_or_zero(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def clip_speed(v: npt.NDArray[np.float64], vmax: float) -> npt.NDArray[np.float64]:
    """
    If vector v is longer than vmax, shrink it to length vmax; otherwise leave it.
    """
    speed = float(np.linalg.norm(v))
    if speed <= vmax or speed == 0:
        return v
    return v * (vmax / speed)


def initial_infielder_layout(
    n_robots: int | None = None,
    field: FieldConfig = DEFAULT_FIELD,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    role_positions = field.infielder_positions()
    default_role_names = list(role_positions.keys())
    default_positions = np.array(list(role_positions.values()), dtype=np.float64)

    if n_robots is None:
        return default_positions, default_role_names

    if n_robots == default_positions.shape[0]:
        return default_positions, default_role_names

    radius = 0.75 * min(field.infield_x_limit, field.infield_y_max)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_robots, endpoint=False, dtype=np.float64)
    positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    positions = clip_positions_to_infield(positions, field)
    role_names = [f"Robot {i}" for i in range(n_robots)]
    return positions.astype(np.float64, copy=False), role_names


def clip_positions_to_infield(
    positions: npt.NDArray[np.float64],
    field: FieldConfig = DEFAULT_FIELD,
) -> npt.NDArray[np.float64]:
    positions[:, 0] = np.clip(positions[:, 0], -field.infield_x_limit, field.infield_x_limit)
    positions[:, 1] = np.clip(positions[:, 1], field.infield_y_min, field.infield_y_max)
    return positions


def make_state(
    n_robots: int | None = None,
    field: FieldConfig = DEFAULT_FIELD,
) -> SimState:
    pos, role_names = initial_infielder_layout(n_robots=n_robots, field=field)
    vel = np.zeros_like(pos)
    return SimState(
        positions=pos,
        velocities=vel,
        home_positions=pos.copy(),
        ball=BallState(
            p0=np.zeros(2, dtype=np.float64),
            v0=np.zeros(2, dtype=np.float64),
            a=np.zeros(2, dtype=np.float64),
            active=False,
        ),
        runner=make_runner(field=field),
        role_names=role_names,
        neighbors=neighbor_indices_ring(pos.shape[0]),
    )


def make_play_state() -> PlayState:
    return PlayState()


def step_formation(
    state: SimState,
    field: FieldConfig = DEFAULT_FIELD,
    dyn: DynamicsConfig = DEFAULT_DYNAMICS,
) -> None:
    raw_command = formation_control_input(state.positions, state.neighbors, dyn.formation_gain)
    clipped = np.zeros_like(raw_command)
    for idx in range(raw_command.shape[0]):
        clipped[idx] = clip_speed(raw_command[idx], dyn.vmax)
    state.velocities = clipped
    state.positions += state.velocities * dyn.dt
    state.positions[:, 0] = np.clip(state.positions[:, 0], field.x_min, field.x_max)
    state.positions[:, 1] = np.clip(state.positions[:, 1], field.y_min, field.y_max)


def current_ball_pos(state: SimState, t: float) -> npt.NDArray[np.float64]:
    if not state.ball.active:
        return state.ball.p0.copy()
    time_since_start = t - state.ball.t0
    return ball_position(time_since_start, state.ball.p0, state.ball.v0, state.ball.a)


def current_force_base_name(
    state: SimState,
    field: FieldConfig = DEFAULT_FIELD,
) -> str | None:
    return current_target_base_name(state.runner, field=field)


def current_force_base_position(
    state: SimState,
    field: FieldConfig = DEFAULT_FIELD,
) -> npt.NDArray[np.float64] | None:
    return current_target_base_position(state.runner, field=field)


def covering_baseman_idx(
    state: SimState,
    base_name: str | None,
    field: FieldConfig = DEFAULT_FIELD,
) -> int | None:
    if base_name is None:
        return None
    role_name = field.covering_role_for_base(base_name)
    if role_name is None:
        return None
    for idx, candidate in enumerate(state.role_names):
        if candidate == role_name:
            return idx
    return None


def plan_current_fielding_decision(
    state: SimState,
    t: float,
    *,
    vmax: float = DEFAULT_DYNAMICS.vmax,
    horizon_s: float = 6.0,
    sample_dt_s: float = DEFAULT_DYNAMICS.dt,
    ownership_penalty_s: float = 0.75,
    ownership_override_margin_s: float = 0.50,
) -> FieldingDecision | None:
    if not state.ball.active:
        return None
    return plan_fielding_assignment(
        state.positions,
        state.home_positions,
        t_now=t,
        ball_t0=state.ball.t0,
        ball_p0=state.ball.p0,
        ball_v0=state.ball.v0,
        ball_a=state.ball.a,
        vmax=vmax,
        horizon_s=horizon_s,
        sample_dt_s=sample_dt_s,
        ownership_penalty_s=ownership_penalty_s,
        ownership_override_margin_s=ownership_override_margin_s,
    )


def launch_ground_ball(
    state: SimState,
    t: float,
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    decel_magnitude: float,
    vmax: float | None = None,
    field: FieldConfig = DEFAULT_FIELD,
) -> None:
    v0 = np.asarray(v0, dtype=np.float64)
    horizontal_part = np.array([v0[0], 0.0], dtype=np.float64)
    horizontal_len = float(np.linalg.norm(horizontal_part))
    if horizontal_len > 1e-6:
        acceleration = -(decel_magnitude / horizontal_len) * horizontal_part
    else:
        acceleration = np.zeros(2, dtype=np.float64)

    state.ball = BallState(
        p0=np.asarray(p0, dtype=np.float64),
        v0=v0,
        a=acceleration,
        t0=t,
        active=True,
    )
    start_runner(state.runner, field=field)

    speed_limit = DEFAULT_DYNAMICS.vmax if vmax is None else vmax
    decision = plan_current_fielding_decision(
        state,
        t,
        vmax=speed_limit,
        horizon_s=6.0,
        sample_dt_s=DEFAULT_DYNAMICS.dt,
    )
    state.primary_idx = None if decision is None else decision.primary_idx


def _freeze_ball(state: SimState, t: float, position: npt.NDArray[np.float64]) -> None:
    state.ball = BallState(
        p0=np.asarray(position, dtype=np.float64),
        v0=np.zeros(2, dtype=np.float64),
        a=np.zeros(2, dtype=np.float64),
        t0=t,
        active=False,
    )


def _set_live_status(
    state: SimState,
    play: PlayState,
    *,
    holder_idx: int | None = None,
    force_base_name: str | None = None,
) -> None:
    if play.resolved:
        return
    if holder_idx is not None and force_base_name is not None:
        holder_name = state.role_names[holder_idx]
        if play.pass_in_flight and play.selected_relay_partner is not None:
            relay_name = state.role_names[play.selected_relay_partner]
            display_base_name = play.relay_target_base or force_base_name
            play.result_text = f"{holder_name} throws to {relay_name} covering {display_base_name}"
        elif play.selected_relay_partner is not None and not play.relay_used:
            relay_name = state.role_names[play.selected_relay_partner]
            play.result_text = f"Fielded by {holder_name}, looking for {relay_name} at {force_base_name}"
        else:
            play.result_text = f"Fielded by {holder_name}, racing to {force_base_name}"
        return
    if state.runner.active:
        force_base = current_force_base_name(state)
        if force_base is not None:
            play.result_text = f"Ball in play: runner racing to {force_base}"


def _maybe_replan_relay(
    state: SimState,
    play: PlayState,
    *,
    method: str,
    vmax: float,
    pass_speed: float,
    tau_handoff: float,
    field: FieldConfig,
) -> None:
    if play.holder_idx is None or method != "ours_relay":
        play.relay_partner = None
        play.relay_target_base = None
        return
    if play.pass_in_flight:
        return
    if play.handoff_done:
        play.relay_partner = None
        play.relay_target_base = current_force_base_name(state, field=field)
        return

    force_base_name = current_force_base_name(state, field=field)
    force_base_pos = current_force_base_position(state, field=field)
    baseman_idx = covering_baseman_idx(state, force_base_name, field=field)
    if force_base_name is None or force_base_pos is None:
        play.relay_partner = None
        play.relay_target_base = None
        return

    if play.relay_target_base == force_base_name and (play.relay_partner is not None or play.handoff_done):
        return

    _, relay_partner = delivery_costs(
        play.holder_idx,
        state.positions,
        force_base_pos,
        vmax,
        first_leg_speed=pass_speed,
        tau_handoff=tau_handoff,
        candidate_indices=(() if baseman_idx is None else (baseman_idx,)),
    )
    play.relay_partner = relay_partner
    play.selected_relay_partner = relay_partner
    play.relay_target_base = force_base_name
    play.pass_in_flight = False
    play.pass_start_time = None
    play.pass_arrival_time = None
    play.pass_start_pos = None
    play.pass_end_pos = None
    play.handoff_done = False
    play.handoff_timer = 0.0


def current_controlled_ball_position(
    state: SimState,
    play: PlayState,
    simulation_time: float,
) -> npt.NDArray[np.float64]:
    if state.ball.active:
        return current_ball_pos(state, simulation_time)
    if (
        play.pass_in_flight
        and play.pass_start_time is not None
        and play.pass_arrival_time is not None
        and play.pass_start_pos is not None
        and play.pass_end_pos is not None
    ):
        duration = max(play.pass_arrival_time - play.pass_start_time, 1e-9)
        alpha = np.clip((simulation_time - play.pass_start_time) / duration, 0.0, 1.0)
        return play.pass_start_pos + alpha * (play.pass_end_pos - play.pass_start_pos)
    if play.has_ball and play.holder_idx is not None:
        return state.positions[play.holder_idx].copy()
    return state.ball.p0.copy()


def step_live_play(
    state: SimState,
    play: PlayState,
    simulation_time: float,
    *,
    method: str = "ours_relay",
    field: FieldConfig = DEFAULT_FIELD,
    vmax: float = DEFAULT_DYNAMICS.vmax,
    formation_gain: float = DEFAULT_DYNAMICS.formation_gain,
    dt: float = DEFAULT_DYNAMICS.dt,
    intercept_radius: float = 0.5,
    base_radius: float = 0.75,
    handoff_radius: float = 0.75,
    tau_handoff: float = 0.0,
    pass_speed: float = DEFAULT_DYNAMICS.pass_speed,
) -> None:
    if play.resolved:
        return

    state.velocities[:] = 0.0

    if play.has_ball and play.holder_idx is not None:
        holder_idx = play.holder_idx
        _maybe_replan_relay(
            state,
            play,
            method=method,
            vmax=vmax,
            pass_speed=pass_speed,
            tau_handoff=tau_handoff,
            field=field,
        )
        force_base_name = current_force_base_name(state, field=field)
        force_base_pos = current_force_base_position(state, field=field)
        _set_live_status(state, play, holder_idx=holder_idx, force_base_name=force_base_name)

        if force_base_name is None or force_base_pos is None:
            play.runner_scored = True
            play.total_time = simulation_time
            play.delivery_time = (
                simulation_time - play.intercept_time if play.intercept_time is not None else float("nan")
            )
            play.result_text = "Runner scores"
            return

        if play.relay_partner is not None and not play.handoff_done:
            relay_target = state.positions[play.relay_partner].copy()
            if not play.pass_in_flight:
                play.pass_in_flight = True
                play.pass_start_time = simulation_time
                play.pass_start_pos = state.positions[holder_idx].copy()
                play.pass_end_pos = relay_target
                pass_distance = float(np.linalg.norm(play.pass_end_pos - play.pass_start_pos))
                flight_time = pass_distance / max(1e-9, float(pass_speed))
                play.pass_arrival_time = simulation_time + max(0.0, float(tau_handoff)) + flight_time

            reached_base_name = advance_runner(
                state.runner,
                dt=dt,
                speed=vmax,
                field=field,
                touch_radius=base_radius,
            )
            if reached_base_name is not None:
                if state.runner.scored:
                    play.runner_scored = True
                    play.total_time = simulation_time
                    play.delivery_time = (
                        simulation_time - play.intercept_time
                        if play.intercept_time is not None
                        else float("nan")
                    )
                    play.result_text = "Runner scores"
                    return
                play.result_text = f"Runner advances to {reached_base_name}"
                play.relay_target_base = None

            if play.pass_arrival_time is not None and simulation_time >= play.pass_arrival_time:
                play.holder_idx = play.relay_partner
                play.relay_used = True
                play.handoff_done = True
                play.pass_in_flight = False
                play.pass_start_time = None
                play.pass_arrival_time = None
                play.pass_start_pos = None
                play.pass_end_pos = None
                play.relay_partner = None
                play.relay_target_base = None
                _freeze_ball(state, simulation_time, state.positions[play.holder_idx].copy())
            return

        direction = _unit_or_zero(force_base_pos - state.positions[holder_idx])
        state.velocities[holder_idx] = direction * vmax
        state.positions += state.velocities * dt
        _clip_to_field(state.positions, field)

        reached_base_name = advance_runner(
            state.runner,
            dt=dt,
            speed=vmax,
            field=field,
            touch_radius=base_radius,
        )
        if reached_base_name is not None:
            if state.runner.scored:
                play.runner_scored = True
                play.total_time = simulation_time
                play.delivery_time = (
                    simulation_time - play.intercept_time
                    if play.intercept_time is not None
                    else float("nan")
                )
                play.result_text = "Runner scores"
                return
            play.result_text = f"Runner advances to {reached_base_name}"
            play.relay_target_base = None
            if force_base_name != reached_base_name:
                holder_idx = play.holder_idx

        current_force_name = current_force_base_name(state, field=field)
        if (
            current_force_name == force_base_name
            and float(np.linalg.norm(state.positions[holder_idx] - force_base_pos)) <= base_radius
        ):
            mark_runner_out(state.runner)
            play.out_recorded = True
            play.total_time = simulation_time
            play.delivery_time = (
                simulation_time - play.intercept_time if play.intercept_time is not None else float("nan")
            )
            play.result_text = f"Out at {force_base_name} by {state.role_names[holder_idx]}"
        return

    if not state.ball.active or state.primary_idx is None:
        _set_live_status(state, play)
        reached_base_name = advance_runner(
            state.runner,
            dt=dt,
            speed=vmax,
            field=field,
            touch_radius=base_radius,
        )
        if reached_base_name is not None:
            if state.runner.scored:
                play.runner_scored = True
                play.total_time = simulation_time
                play.result_text = "Runner scores"
                return
            play.result_text = f"Runner advances to {reached_base_name}"
        return

    ball_xy = current_ball_pos(state, simulation_time)
    primary = state.primary_idx
    pursuit_target = ball_xy

    if method != "nearest_direct":
        decision = plan_current_fielding_decision(state, simulation_time, vmax=vmax, sample_dt_s=dt)
        if decision is not None:
            state.primary_idx = decision.primary_idx
            primary = decision.primary_idx
            pursuit_target = decision.primary_intercept_point

    direction = _unit_or_zero(pursuit_target - state.positions[primary])
    state.velocities[primary] = direction * vmax

    raw_command = formation_control_input(state.positions, state.neighbors, formation_gain)
    for idx in range(state.positions.shape[0]):
        if idx == primary:
            continue
        state.velocities[idx] = clip_speed(raw_command[idx], vmax)

    state.positions += state.velocities * dt
    _clip_to_field(state.positions, field)

    reached_base_name = advance_runner(
        state.runner,
        dt=dt,
        speed=vmax,
        field=field,
        touch_radius=base_radius,
    )
    if reached_base_name is not None:
        if state.runner.scored:
            play.runner_scored = True
            play.total_time = simulation_time
            play.result_text = "Runner scores"
            return
        play.result_text = f"Runner advances to {reached_base_name}"
    else:
        _set_live_status(state, play)

    if float(np.linalg.norm(state.positions[primary] - ball_xy)) <= intercept_radius:
        play.intercept_time = simulation_time
        play.has_ball = True
        play.holder_idx = primary
        _freeze_ball(state, simulation_time, state.positions[primary].copy())
        force_base_name = current_force_base_name(state, field=field)
        _set_live_status(state, play, holder_idx=primary, force_base_name=force_base_name)


def run_episode(
    cfg: EpisodeConfig,
    *,
    method: str = "ours_relay",
    field: FieldConfig = DEFAULT_FIELD,
) -> dict[str, float | int | bool | None]:
    """
    Run one episode and return scalar metrics for tables/CSV.
    """
    if method not in {"nearest_direct", "ours_direct", "ours_relay"}:
        raise ValueError(f"Unknown method: {method}")

    state = make_state(n_robots=cfg.n_robots, field=field)
    state.neighbors = neighbor_indices_ring(cfg.n_robots)
    state.velocities = np.zeros_like(state.positions)
    play = make_play_state()

    t = 0.0
    while t < cfg.max_time and not play.resolved:
        t += cfg.dt

        if not play.launched and t < cfg.launch_time:
            raw_command = formation_control_input(
                state.positions, state.neighbors, cfg.formation_gain
            )
            state.velocities = np.zeros_like(state.positions)
            for idx in range(state.positions.shape[0]):
                state.velocities[idx] = clip_speed(raw_command[idx], cfg.vmax)
            state.positions += state.velocities * cfg.dt
            _clip_to_field(state.positions, field)
            continue

        if not play.launched:
            launch_ground_ball(
                state,
                t,
                p0=np.array(cfg.ball_p0, dtype=np.float64),
                v0=np.array(cfg.ball_v0, dtype=np.float64),
                decel_magnitude=cfg.ball_decel,
                vmax=cfg.vmax,
                field=field,
            )
            play.launched = True
            play.result_text = "Ball in play: runner racing to first"

            if method == "nearest_direct":
                ball_now = current_ball_pos(state, t)
                dists = np.linalg.norm(state.positions - ball_now[None, :], axis=1)
                state.primary_idx = int(np.argmin(dists))

        step_live_play(
            state,
            play,
            t,
            method=method,
            field=field,
            vmax=cfg.vmax,
            formation_gain=cfg.formation_gain,
            dt=cfg.dt,
            intercept_radius=cfg.intercept_radius,
            base_radius=cfg.base_radius,
            handoff_radius=cfg.handoff_radius,
            tau_handoff=cfg.tau_handoff,
            pass_speed=cfg.pass_speed,
        )

    success = play.out_recorded
    return {
        "success": bool(success),
        "out_recorded": bool(play.out_recorded),
        "runner_scored": bool(play.runner_scored),
        "t_intercept": float(play.intercept_time) if play.intercept_time is not None else float("nan"),
        "t_delivery": float(play.delivery_time) if play.delivery_time is not None else float("nan"),
        "t_total": float(play.total_time) if play.total_time is not None else float("nan"),
        "primary_idx": int(state.primary_idx) if state.primary_idx is not None else -1,
        "relay_used": bool(play.relay_used),
        "relay_partner": (
            int(play.selected_relay_partner) if play.selected_relay_partner is not None else -1
        ),
    }
