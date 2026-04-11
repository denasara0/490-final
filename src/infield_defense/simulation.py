from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from infield_defense.ball import ball_position
from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD, DynamicsConfig, FieldConfig
from infield_defense.coordination import select_primary_fielder
from infield_defense.formation import formation_control_input, neighbor_indices_ring

@dataclass
class BallState:
    p0: npt.NDArray[np.float64]  # Starting position when this segment began
    v0: npt.NDArray[np.float64]  # Starting velocity
    a: npt.NDArray[np.float64]  # Constant acceleration (slowdown along x in our demo)
    t0: float = 0.0  # Simulation clock time when this segment started
    active: bool = False  # False = no ground ball in play yet


@dataclass
class SimState:
    positions: npt.NDArray[np.float64]  # Shape (N, 2)
    velocities: npt.NDArray[np.float64]  # Shape (N, 2)
    home_positions: npt.NDArray[np.float64]  # Shape (N, 2), default defensive spots
    ball: BallState
    primary_idx: int | None = None  # Which robot is chosen to chase the ball; None before assignment
    role_names: list[str] = field(default_factory=list)
    neighbors: list[list[int]] = field(default_factory=list)  # From formation.neighbor_indices_ring


def clip_speed(v: npt.NDArray[np.float64], vmax: float) -> npt.NDArray[np.float64]:
    """
    If vector v is longer than vmax, shrink it to length vmax; otherwise leave it.

    Robots cannot move faster than vmax, so we cap velocity commands here.
    """
    speed = float(np.linalg.norm(v))
    if speed <= vmax or speed == 0:
        return v
    return v * (vmax / speed)


def initial_infielder_layout(
    field: FieldConfig = DEFAULT_FIELD,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    role_positions = field.infielder_positions()
    role_names = list(role_positions.keys())
    positions = np.array(list(role_positions.values()), dtype=np.float64)
    return positions, role_names


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
    pos, role_names = initial_infielder_layout(field)
    if n_robots is not None and n_robots != pos.shape[0]:
        raise ValueError(f"This demo renders exactly {pos.shape[0]} robots (received {n_robots}).")
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
        role_names=role_names,
        neighbors=neighbor_indices_ring(pos.shape[0]),
    )


def step_formation(
    state: SimState,
    field: FieldConfig = DEFAULT_FIELD,
    dyn: DynamicsConfig = DEFAULT_DYNAMICS,
) -> None:
    raw_command = formation_control_input(state.positions, state.neighbors, dyn.formation_gain)
    state.velocities = clip_speed(raw_command, dyn.vmax)
    state.positions += state.velocities * dyn.dt
    state.positions[:, 0] = np.clip(state.positions[:, 0], field.x_min, field.x_max)
    state.positions[:, 1] = np.clip(state.positions[:, 1], field.y_min, field.y_max)


def current_ball_pos(state: SimState, t: float) -> npt.NDArray[np.float64]:
    if not state.ball.active:
        return state.ball.p0.copy()
    time_since_start = t - state.ball.t0
    return ball_position(time_since_start, state.ball.p0, state.ball.v0, state.ball.a)


def launch_ground_ball(
    state: SimState,
    t: float,
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    decel_magnitude: float,
    vmax: float | None = None,
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

    speed_limit = DEFAULT_DYNAMICS.vmax if vmax is None else vmax
    ball_now = current_ball_pos(state, t)
    state.primary_idx = select_primary_fielder(state.positions, ball_now, speed_limit)
