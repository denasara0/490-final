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
from infield_defense.coordination import delivery_costs, select_primary_fielder
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


@dataclass(frozen=True)
class EpisodeConfig:
    """
    Parameters for a single headless simulation episode, for repeated-trial evaluation.

    This intentionally mirrors the report's evaluation protocol: randomized ball launches,
    time-to-intercept, and delivery time with optional relay handoff delay.
    """

    n_robots: int = 5
    vmax: float = DEFAULT_DYNAMICS.vmax
    formation_gain: float = DEFAULT_DYNAMICS.formation_gain
    dt: float = DEFAULT_DYNAMICS.dt
    launch_time: float = 0.5

    # Ball initial conditions (can be randomized per trial by caller if desired).
    ball_p0: tuple[float, float] = (-30.0, -5.0)
    ball_v0: tuple[float, float] = (14.0, 2.0)
    ball_decel: float = DEFAULT_DYNAMICS.ball_decel

    # Capture + delivery
    intercept_radius: float = 0.5
    base_radius: float = 0.75
    handoff_radius: float = 0.75
    tau_handoff: float = 0.0

    # Safety
    max_time: float = 20.0


def _clip_to_field(positions: npt.NDArray[np.float64], field: FieldConfig) -> None:
    positions[:, 0] = np.clip(positions[:, 0], field.x_min, field.x_max)
    positions[:, 1] = np.clip(positions[:, 1], field.y_min, field.y_max)


def _unit_or_zero(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def run_episode(
    cfg: EpisodeConfig,
    *,
    method: str = "ours_relay",
    field: FieldConfig = DEFAULT_FIELD,
) -> dict[str, float | int | bool | None]:
    """
    Run one episode and return scalar metrics for tables/CSV.

    Methods:
        - "nearest_direct": primary = nearest-to-ball-at-detection, delivery forced direct
        - "ours_direct": primary = argmin(distance/vmax) at detection, delivery forced direct
        - "ours_relay": primary = argmin(distance/vmax) at detection, delivery uses relay_costs
    """
    if method not in {"nearest_direct", "ours_direct", "ours_relay"}:
        raise ValueError(f"Unknown method: {method}")

    # Build state (reuse existing formation neighbors + arc initializer).
    state = make_state(n_robots=cfg.n_robots)
    state.neighbors = neighbor_indices_ring(cfg.n_robots)
    state.velocities = np.zeros_like(state.positions)

    base_pos = np.array(field.first_base, dtype=np.float64)
    t = 0.0
    launched = False
    intercept_time: float | None = None
    delivery_time: float | None = None
    total_time: float | None = None
    relay_partner: int | None = None

    # Phase bookkeeping
    has_ball = False
    holder_idx: int | None = None
    handoff_done = False
    handoff_timer = 0.0

    while t < cfg.max_time:
        t += cfg.dt

        # Pre-launch: hold formation.
        if not launched and t < cfg.launch_time:
            raw_command = formation_control_input(state.positions, state.neighbors, cfg.formation_gain)
            state.velocities = clip_speed(raw_command, cfg.vmax)
            state.positions += state.velocities * cfg.dt
            _clip_to_field(state.positions, field)
            continue

        # Launch ball once.
        if not launched:
            launch_ground_ball(
                state,
                t,
                p0=np.array(cfg.ball_p0, dtype=np.float64),
                v0=np.array(cfg.ball_v0, dtype=np.float64),
                decel_magnitude=cfg.ball_decel,
                vmax=cfg.vmax,
            )
            launched = True

            # Override primary selection for nearest baseline.
            if method == "nearest_direct":
                ball_now = current_ball_pos(state, t)
                dists = np.linalg.norm(state.positions - ball_now[None, :], axis=1)
                state.primary_idx = int(np.argmin(dists))

        # If already captured, run delivery.
        if has_ball and holder_idx is not None:
            # Decide relay partner once (at capture time) for methods that allow relay.
            if intercept_time is not None and total_time is None and delivery_time is None:
                if method == "ours_relay":
                    _, relay_partner = delivery_costs(
                        holder_idx,
                        state.positions,
                        base_pos,
                        cfg.vmax,
                        tau_handoff=cfg.tau_handoff,
                    )
                else:
                    relay_partner = None

            # If relaying and handoff not done, holder runs to relay partner.
            if relay_partner is not None and not handoff_done:
                target = state.positions[relay_partner]
                direction = _unit_or_zero(target - state.positions[holder_idx])
                state.velocities[:] = 0.0
                state.velocities[holder_idx] = direction * cfg.vmax
                state.positions += state.velocities * cfg.dt
                _clip_to_field(state.positions, field)

                if float(np.linalg.norm(state.positions[holder_idx] - state.positions[relay_partner])) <= cfg.handoff_radius:
                    # Simulate handoff delay.
                    handoff_timer += cfg.dt
                    if handoff_timer >= cfg.tau_handoff:
                        holder_idx = relay_partner
                        handoff_done = True
                continue

            # Direct leg to base (either no relay, or after handoff).
            direction = _unit_or_zero(base_pos - state.positions[holder_idx])
            state.velocities[:] = 0.0
            state.velocities[holder_idx] = direction * cfg.vmax
            state.positions += state.velocities * cfg.dt
            _clip_to_field(state.positions, field)

            if float(np.linalg.norm(state.positions[holder_idx] - base_pos)) <= cfg.base_radius:
                # Finish delivery.
                if intercept_time is None:
                    # Shouldn't happen, but keep metrics consistent.
                    intercept_time = t
                delivery_time = t - intercept_time
                total_time = t
                break

            continue

        # Otherwise: chase ball (primary) + formation (others).
        if state.ball.active and state.primary_idx is not None:
            ball_xy = current_ball_pos(state, t)
            primary = state.primary_idx

            # Primary heads toward current ball estimate.
            direction = _unit_or_zero(ball_xy - state.positions[primary])
            state.velocities[primary] = direction * cfg.vmax

            # Others hold formation.
            for i in range(state.positions.shape[0]):
                if i == primary:
                    continue
                # single-step formation command for others (cheap; avoids subset remap here)
                # (This is a simulation convenience; it does not change the "distributed" math.)
                pass
            raw_command = formation_control_input(state.positions, state.neighbors, cfg.formation_gain)
            for i in range(state.positions.shape[0]):
                if i == primary:
                    continue
                state.velocities[i] = raw_command[i]
                state.velocities[i] = clip_speed(state.velocities[i], cfg.vmax)

            state.positions += state.velocities * cfg.dt
            _clip_to_field(state.positions, field)

            # Check capture.
            if float(np.linalg.norm(state.positions[primary] - ball_xy)) <= cfg.intercept_radius:
                intercept_time = t
                has_ball = True
                holder_idx = primary

    success = total_time is not None and intercept_time is not None and delivery_time is not None
    return {
        "success": bool(success),
        "t_intercept": float(intercept_time) if intercept_time is not None else float("nan"),
        "t_delivery": float(delivery_time) if delivery_time is not None else float("nan"),
        "t_total": float(total_time) if total_time is not None else float("nan"),
        "primary_idx": int(state.primary_idx) if state.primary_idx is not None else -1,
        "relay_used": bool(relay_partner is not None),
        "relay_partner": int(relay_partner) if relay_partner is not None else -1,
    }

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
