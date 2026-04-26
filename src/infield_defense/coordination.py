from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from infield_defense.ball import ball_position, ball_velocity


@dataclass(frozen=True)
class FieldingDecision:
    primary_idx: int
    trajectory_owner_idx: int
    intercept_times: npt.NDArray[np.float64]
    intercept_points: npt.NDArray[np.float64]
    fielding_scores: npt.NDArray[np.float64]

    @property
    def primary_intercept_point(self) -> npt.NDArray[np.float64]:
        return self.intercept_points[self.primary_idx]


def interception_time_estimate(
    robot_pos: npt.NDArray[np.float64],
    ball_pos: npt.NDArray[np.float64],
    vmax: float,
) -> float:
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    distance = float(np.linalg.norm(robot_pos - ball_pos))
    if vmax <= 0:
        return float("inf")
    return distance / vmax


def interception_plan_to_moving_ball(
    robot_pos: npt.NDArray[np.float64],
    *,
    t_now: float,
    ball_t0: float,
    ball_p0: npt.NDArray[np.float64],
    ball_v0: npt.NDArray[np.float64],
    ball_a: npt.NDArray[np.float64],
    vmax: float,
    horizon_s: float = 6.0,
    sample_dt_s: float = 0.05,
) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Return the earliest feasible intercept time and point for a moving ball.
    """
    ball_p0 = np.asarray(ball_p0, dtype=np.float64)
    ball_v0 = np.asarray(ball_v0, dtype=np.float64)
    ball_a = np.asarray(ball_a, dtype=np.float64)
    robot_pos = np.asarray(robot_pos, dtype=np.float64)

    def point_at_tau(tau: float) -> npt.NDArray[np.float64]:
        t_abs = float(t_now) + float(tau)
        t_ball = t_abs - float(ball_t0)
        return ball_position(t_ball, ball_p0, ball_v0, ball_a)

    if vmax <= 0:
        return float("inf"), point_at_tau(0.0)

    horizon_s = max(0.0, float(horizon_s))
    sample_dt_s = max(1e-3, float(sample_dt_s))

    # f(tau) <= 0 means the ball position at tau is reachable by time tau.
    def f(tau: float) -> float:
        p_ball = point_at_tau(tau)
        return float(np.linalg.norm(robot_pos - p_ball) - vmax * float(tau))

    if f(0.0) <= 0.0:
        return 0.0, point_at_tau(0.0)

    n_steps = int(np.ceil(horizon_s / sample_dt_s))
    t_lo = 0.0
    t_hi = None
    for k in range(1, n_steps + 1):
        t = min(horizon_s, k * sample_dt_s)
        if f(t) <= 0.0:
            t_hi = t
            break
        t_lo = t

    if t_hi is None:
        return float("inf"), point_at_tau(0.0)

    lo, hi = t_lo, t_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi, point_at_tau(hi)


def interception_time_to_moving_ball(
    robot_pos: npt.NDArray[np.float64],
    *,
    t_now: float,
    ball_t0: float,
    ball_p0: npt.NDArray[np.float64],
    ball_v0: npt.NDArray[np.float64],
    ball_a: npt.NDArray[np.float64],
    vmax: float,
    horizon_s: float = 6.0,
    sample_dt_s: float = 0.05,
) -> float:
    """
    Estimate the earliest feasible intercept time for a moving ball under a speed bound.

    We assume a holonomic point robot that can move in any direction with speed <= vmax.
    The robot can intercept at elapsed time tau >= 0 if:

        ||p_robot - p_ball(t_now + tau)|| <= vmax * tau

    We return the smallest tau (seconds) within [0, horizon_s] that satisfies this.
    If no feasible intercept exists in the horizon, return +inf.
    """
    tau, _ = interception_plan_to_moving_ball(
        robot_pos,
        t_now=t_now,
        ball_t0=ball_t0,
        ball_p0=ball_p0,
        ball_v0=ball_v0,
        ball_a=ball_a,
        vmax=vmax,
        horizon_s=horizon_s,
        sample_dt_s=sample_dt_s,
    )
    return tau


def select_primary_fielder(
    positions: npt.NDArray[np.float64],
    ball_pos: npt.NDArray[np.float64],
    vmax: float,
) -> int:
    n = positions.shape[0]
    times = []
    for i in range(n):
        t_i = interception_time_estimate(positions[i], ball_pos, vmax)
        times.append(t_i)
    return int(np.argmin(np.array(times)))


def select_primary_fielder_predicted(
    positions: npt.NDArray[np.float64],
    *,
    t_now: float,
    ball_t0: float,
    ball_p0: npt.NDArray[np.float64],
    ball_v0: npt.NDArray[np.float64],
    ball_a: npt.NDArray[np.float64],
    vmax: float,
    horizon_s: float = 6.0,
    sample_dt_s: float = 0.05,
) -> int:
    """
    Primary selection using a moving-ball intercept-time bid (earliest feasible intercept).
    """
    times = [
        interception_time_to_moving_ball(
            positions[i],
            t_now=t_now,
            ball_t0=ball_t0,
            ball_p0=ball_p0,
            ball_v0=ball_v0,
            ball_a=ball_a,
            vmax=vmax,
            horizon_s=horizon_s,
            sample_dt_s=sample_dt_s,
        )
        for i in range(positions.shape[0])
    ]
    return int(np.argmin(np.asarray(times, dtype=np.float64)))


def trajectory_ray_distance(
    point: npt.NDArray[np.float64],
    ray_origin: npt.NDArray[np.float64],
    ray_direction: npt.NDArray[np.float64],
) -> float:
    """
    Distance from a point to the forward trajectory ray of the ball.
    """
    point = np.asarray(point, dtype=np.float64)
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_direction = np.asarray(ray_direction, dtype=np.float64)

    direction_norm = float(np.linalg.norm(ray_direction))
    offset = point - ray_origin
    if direction_norm <= 1e-12:
        return float(np.linalg.norm(offset))

    direction_unit = ray_direction / direction_norm
    projection = float(np.dot(offset, direction_unit))
    if projection <= 0.0:
        return float(np.linalg.norm(offset))
    perpendicular = offset - projection * direction_unit
    return float(np.linalg.norm(perpendicular))


def select_trajectory_owner(
    home_positions: npt.NDArray[np.float64],
    *,
    ball_pos: npt.NDArray[np.float64],
    ball_direction: npt.NDArray[np.float64],
) -> int:
    distances = [
        trajectory_ray_distance(home_positions[i], ball_pos, ball_direction)
        for i in range(home_positions.shape[0])
    ]
    return int(np.argmin(np.asarray(distances, dtype=np.float64)))


def plan_fielding_assignment(
    positions: npt.NDArray[np.float64],
    home_positions: npt.NDArray[np.float64],
    *,
    t_now: float,
    ball_t0: float,
    ball_p0: npt.NDArray[np.float64],
    ball_v0: npt.NDArray[np.float64],
    ball_a: npt.NDArray[np.float64],
    vmax: float,
    horizon_s: float = 6.0,
    sample_dt_s: float = 0.05,
    ownership_penalty_s: float = 0.75,
    ownership_override_margin_s: float = 0.50,
) -> FieldingDecision:
    """
    Score fielders using moving-ball intercept time plus a trajectory ownership bias.
    """
    positions = np.asarray(positions, dtype=np.float64)
    home_positions = np.asarray(home_positions, dtype=np.float64)
    ball_p0 = np.asarray(ball_p0, dtype=np.float64)
    ball_v0 = np.asarray(ball_v0, dtype=np.float64)
    ball_a = np.asarray(ball_a, dtype=np.float64)

    t_ball = float(t_now) - float(ball_t0)
    current_ball_pos = ball_position(t_ball, ball_p0, ball_v0, ball_a)
    current_ball_velocity = ball_velocity(t_ball, ball_v0, ball_a)
    owner_idx = select_trajectory_owner(
        home_positions,
        ball_pos=current_ball_pos,
        ball_direction=current_ball_velocity,
    )

    n_robots = positions.shape[0]
    intercept_times = np.full(n_robots, float("inf"), dtype=np.float64)
    intercept_points = np.tile(current_ball_pos, (n_robots, 1)).astype(np.float64, copy=False)

    for robot_idx in range(n_robots):
        tau, intercept_point = interception_plan_to_moving_ball(
            positions[robot_idx],
            t_now=t_now,
            ball_t0=ball_t0,
            ball_p0=ball_p0,
            ball_v0=ball_v0,
            ball_a=ball_a,
            vmax=vmax,
            horizon_s=horizon_s,
            sample_dt_s=sample_dt_s,
        )
        intercept_times[robot_idx] = tau
        intercept_points[robot_idx] = intercept_point

    if bool(np.all(np.isinf(intercept_times))):
        intercept_times = np.asarray(
            [
                interception_time_estimate(positions[i], current_ball_pos, vmax)
                for i in range(n_robots)
            ],
            dtype=np.float64,
        )
        intercept_points = np.tile(current_ball_pos, (n_robots, 1)).astype(np.float64, copy=False)

    fielding_scores = intercept_times.copy()
    non_owner_mask = np.arange(n_robots) != owner_idx
    fielding_scores[non_owner_mask] += max(0.0, float(ownership_penalty_s))

    primary_idx = owner_idx
    owner_time = float(intercept_times[owner_idx])
    if np.isfinite(owner_time):
        challenger_times = intercept_times.copy()
        challenger_times[owner_idx] = float("inf")
        challenger_idx = int(np.argmin(challenger_times))
        challenger_time = float(challenger_times[challenger_idx])
        if (
            np.isfinite(challenger_time)
            and challenger_time + float(ownership_override_margin_s) < owner_time
        ):
            primary_idx = challenger_idx
    else:
        primary_idx = int(np.argmin(fielding_scores))

    return FieldingDecision(
        primary_idx=primary_idx,
        trajectory_owner_idx=owner_idx,
        intercept_times=intercept_times,
        intercept_points=intercept_points,
        fielding_scores=fielding_scores,
    )


def delivery_costs(
    holder_idx: int,
    positions: npt.NDArray[np.float64],
    base_pos: npt.NDArray[np.float64],
    vmax: float,
    *,
    first_leg_speed: float | None = None,
    tau_handoff: float = 0.0,
    candidate_indices: tuple[int, ...] | None = None,
) -> tuple[float, int | None]:
    """
    After someone has the ball, estimate the fastest way to get it to base.

    compare:
        * Direct: holder runs straight to base (one leg).
        * Relay: holder runs to teammate j, teammate runs to base (two legs).

    Args:
        holder_idx: Index of the robot currently holding the ball.
        positions: All robot positions, shape (N, 2).
        base_pos: Target base [x, y].
        vmax: Top speed used for every leg.
        first_leg_speed: Speed for the first leg from holder to teammate.
            Defaults to ``vmax`` when omitted.
        tau_handoff: Fixed delay to represent a handoff/pass at the relay (seconds).
        candidate_indices: Optional subset of teammate indices to consider for relay.

    Returns:
        (best_time, relay_partner_or_none)
        If the second value is None, direct delivery is at least as good as any relay we checked.
        Otherwise it is the index j of the best relay partner.
    """
    holder_pos = positions[holder_idx]

    time_direct = interception_time_estimate(holder_pos, base_pos, vmax)

    relay_speed = vmax if first_leg_speed is None else max(0.0, float(first_leg_speed))
    candidate_set = range(positions.shape[0]) if candidate_indices is None else candidate_indices

    best_relay_time = float("inf")
    best_teammate: int | None = None

    for teammate_j in candidate_set:
        if teammate_j == holder_idx:
            continue
        time_to_teammate = interception_time_estimate(holder_pos, positions[teammate_j], relay_speed)
        time_teammate_to_base = interception_time_estimate(positions[teammate_j], base_pos, vmax)
        time_relay = time_to_teammate + time_teammate_to_base + max(0.0, float(tau_handoff))

        if time_relay < best_relay_time:
            best_relay_time = time_relay
            best_teammate = teammate_j

    if time_direct <= best_relay_time:
        return time_direct, None
    return best_relay_time, best_teammate
