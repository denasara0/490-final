from __future__ import annotations

import numpy as np
import numpy.typing as npt

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

def delivery_costs(
    holder_idx: int,
    positions: npt.NDArray[np.float64],
    base_pos: npt.NDArray[np.float64],
    vmax: float,
) -> tuple[float, int | None]:
    holder_pos = positions[holder_idx]

    time_direct = interception_time_estimate(holder_pos, base_pos, vmax)

    best_relay_time = float("inf")
    best_teammate: int | None = None

    for teammate_j in range(positions.shape[0]):
        if teammate_j == holder_idx:
            continue
        time_to_teammate = interception_time_estimate(holder_pos, positions[teammate_j], vmax)
        time_teammate_to_base = interception_time_estimate(positions[teammate_j], base_pos, vmax)
        time_relay = time_to_teammate + time_teammate_to_base

        if time_relay < best_relay_time:
            best_relay_time = time_relay
            best_teammate = teammate_j

    if time_direct <= best_relay_time:
        return time_direct, None
    return best_relay_time, best_teammate
