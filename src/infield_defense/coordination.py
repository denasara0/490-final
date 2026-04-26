from __future__ import annotations

import numpy as np
import numpy.typing as npt

from infield_defense.ball import ball_position


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
    if vmax <= 0:
        return float("inf")
    horizon_s = max(0.0, float(horizon_s))
    sample_dt_s = max(1e-3, float(sample_dt_s))

    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    ball_p0 = np.asarray(ball_p0, dtype=np.float64)
    ball_v0 = np.asarray(ball_v0, dtype=np.float64)
    ball_a = np.asarray(ball_a, dtype=np.float64)

    # f(tau) <= 0 means the ball position at tau is reachable by time tau.
    def f(tau: float) -> float:
        t_abs = float(t_now) + float(tau)
        t_ball = t_abs - float(ball_t0)
        p_ball = ball_position(t_ball, ball_p0, ball_v0, ball_a)
        return float(np.linalg.norm(robot_pos - p_ball) - vmax * float(tau))

    # Quick exit: if already "at" the ball at current time.
    if f(0.0) <= 0.0:
        return 0.0

    # Find a bracket [t_lo, t_hi] where f crosses from >0 to <=0 by sampling.
    n_steps = int(np.ceil(horizon_s / sample_dt_s))
    t_lo = 0.0
    f_lo = f(t_lo)
    t_hi = None
    f_hi = None
    for k in range(1, n_steps + 1):
        t = min(horizon_s, k * sample_dt_s)
        f_t = f(t)
        if f_t <= 0.0:
            t_hi = t
            f_hi = f_t
            break
        t_lo, f_lo = t, f_t

    if t_hi is None or f_hi is None:
        return float("inf")

    # Binary search for earliest feasible tau in [t_lo, t_hi].
    lo, hi = t_lo, t_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


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

def delivery_costs(
    holder_idx: int,
    positions: npt.NDArray[np.float64],
    base_pos: npt.NDArray[np.float64],
    vmax: float,
    tau_handoff: float = 0.0,
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
        tau_handoff: Fixed delay to represent a handoff/pass at the relay (seconds).

    Returns:
        (best_time, relay_partner_or_none)
        If the second value is None, direct delivery is at least as good as any relay we checked.
        Otherwise it is the index j of the best relay partner.
    """
    holder_pos = positions[holder_idx]

    time_direct = interception_time_estimate(holder_pos, base_pos, vmax)

    best_relay_time = float("inf")
    best_teammate: int | None = None

    for teammate_j in range(positions.shape[0]):
        if teammate_j == holder_idx:
            continue
        time_to_teammate = interception_time_estimate(holder_pos, positions[teammate_j], vmax)
        time_teammate_to_base = interception_time_estimate(positions[teammate_j], base_pos, vmax)
        time_relay = time_to_teammate + time_teammate_to_base + max(0.0, float(tau_handoff))

        if time_relay < best_relay_time:
            best_relay_time = time_relay
            best_teammate = teammate_j

    if time_direct <= best_relay_time:
        return time_direct, None
    return best_relay_time, best_teammate
