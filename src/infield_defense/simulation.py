"""
Simulation state and time stepping.

This file ties together:
    - Robot positions and velocities
    - Ball trajectory parameters
    - One small Euler step: new_position = old_position + velocity * dt

It does not draw graphics; see ``cli.py`` for that.
"""

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
    """Everything needed to predict the ball path until you change these values."""

    p0: npt.NDArray[np.float64]  # Starting position when this segment began
    v0: npt.NDArray[np.float64]  # Starting velocity
    a: npt.NDArray[np.float64]  # Constant acceleration (slowdown along x in our demo)
    t0: float = 0.0  # Simulation clock time when this segment started
    active: bool = False  # False = no ground ball in play yet


@dataclass
class SimState:
    """Full snapshot of the multi-robot + ball world."""

    positions: npt.NDArray[np.float64]  # Shape (N, 2)
    velocities: npt.NDArray[np.float64]  # Shape (N, 2)
    ball: BallState
    primary_idx: int | None = None  # Which robot is chosen to chase the ball; None before assignment
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


def initial_positions_even_arc(
    n: int,
    center: tuple[float, float] = (0.0, 10.0),
    radius: float = 18.0,
    span_deg: float = 100.0,
) -> npt.NDArray[np.float64]:
    """
    Place n robots on a circular arc — handy for a readable default picture.

    You can change center, radius, or span_deg to get different starting shapes.
    """
    cx, cy = center
    angles = np.linspace(
        np.deg2rad(90 + span_deg / 2),
        np.deg2rad(90 - span_deg / 2),
        n,
    )
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def make_state(n_robots: int = 5) -> SimState:
    """Create a fresh simulation with robots on an arc and no active ball."""
    pos = initial_positions_even_arc(n_robots)
    vel = np.zeros_like(pos)
    return SimState(
        positions=pos,
        velocities=vel,
        ball=BallState(
            p0=np.zeros(2, dtype=np.float64),
            v0=np.zeros(2, dtype=np.float64),
            a=np.zeros(2, dtype=np.float64),
            active=False,
        ),
        neighbors=neighbor_indices_ring(n_robots),
    )


def step_formation(
    state: SimState,
    field: FieldConfig = DEFAULT_FIELD,
    dyn: DynamicsConfig = DEFAULT_DYNAMICS,
) -> None:
    """
    Advance the clock by one step while everyone is in "defensive formation" mode.

    Mutates ``state`` in place: updates velocities then positions.
    """
    raw_command = formation_control_input(state.positions, state.neighbors, dyn.formation_gain)
    state.velocities = clip_speed(raw_command, dyn.vmax)
    state.positions += state.velocities * dyn.dt
    # Keep robots inside the field rectangle (simple bounce substitute).
    state.positions[:, 0] = np.clip(state.positions[:, 0], field.x_min, field.x_max)
    state.positions[:, 1] = np.clip(state.positions[:, 1], field.y_min, field.y_max)


def current_ball_pos(state: SimState, t: float) -> npt.NDArray[np.float64]:
    """Ball [x, y] at simulation time t (uses ball.py if the ball is active)."""
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
    """
    Start a new ground ball from rest conditions (p0, v0) and pick the primary fielder.

    Acceleration is chosen so the ball slows down horizontally (very rough "friction").

    Args:
        state: Full simulation state (updated in place).
        t: Current simulation time (stored on the ball so we know how long it has rolled).
        p0, v0: Initial ball position and velocity.
        decel_magnitude: How strong the slowdown is (bigger = stops sooner).
        vmax: Optional override for interception-time math; default comes from config.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    # Only slow the ball along the horizontal part of its motion (simple demo model).
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
