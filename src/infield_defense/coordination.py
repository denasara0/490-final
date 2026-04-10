"""
Decisions about *who* fields the ball and *how* it gets to base.

These functions use simple geometry, not a full path planner:

- **Interception time** (proposal): straight-line distance to the ball divided by top speed.
  That is optimistic (ignores obstacles) but easy to understand and fast to compute.

- **Delivery**: compare "run straight to base" vs "toss to teammate, teammate runs to base"
  using the same straight-line time idea for each leg.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def interception_time_estimate(
    robot_pos: npt.NDArray[np.float64],
    ball_pos: npt.NDArray[np.float64],
    vmax: float,
) -> float:
    """
    Rough guess: time for a robot to reach the ball if it could drive in a straight line at vmax.

    Formula from the proposal: distance / vmax.
    """
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    distance = float(np.linalg.norm(robot_pos - ball_pos))
    if vmax <= 0:
        return float("inf")
    return distance / vmax


def select_primary_fielder(
    positions: npt.NDArray[np.float64],
    ball_pos: npt.NDArray[np.float64],
    vmax: float,
) -> int:
    """
    Pick the robot index with the *smallest* estimated interception time.

    That robot is treated as the primary fielder 
    """
    n = positions.shape[0]
    times = []
    for i in range(n):
        t_i = interception_time_estimate(positions[i], ball_pos, vmax)
        times.append(t_i)
    # argmin returns the index of the minimum value in the list.
    return int(np.argmin(np.array(times)))


def delivery_costs(
    holder_idx: int,
    positions: npt.NDArray[np.float64],
    base_pos: npt.NDArray[np.float64],
    vmax: float,
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

    Returns:
        (best_time, relay_partner_or_none)
        If the second value is None, direct delivery is at least as good as any relay we checked.
        Otherwise it is the index j of the best relay partner.
    """
    holder_pos = positions[holder_idx]

    # Time if the holder runs straight to the base.
    time_direct = interception_time_estimate(holder_pos, base_pos, vmax)

    best_relay_time = float("inf")
    best_teammate: int | None = None

    for teammate_j in range(positions.shape[0]):
        if teammate_j == holder_idx:
            continue
        # Leg 1: holder to teammate. Leg 2: teammate to base.
        time_to_teammate = interception_time_estimate(holder_pos, positions[teammate_j], vmax)
        time_teammate_to_base = interception_time_estimate(positions[teammate_j], base_pos, vmax)
        time_relay = time_to_teammate + time_teammate_to_base

        if time_relay < best_relay_time:
            best_relay_time = time_relay
            best_teammate = teammate_j

    if time_direct <= best_relay_time:
        return time_direct, None
    return best_relay_time, best_teammate
