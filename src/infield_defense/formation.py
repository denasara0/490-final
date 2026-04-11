from __future__ import annotations

import numpy as np
import numpy.typing as npt

def neighbor_indices_ring(n: int) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n
        neighbors.append([left, right])
    return neighbors


def formation_control_input(
    positions: npt.NDArray[np.float64],
    neighbors: list[list[int]],
    k: float,
) -> npt.NDArray[np.float64]:
    n_robots = positions.shape[0]
    command = np.zeros_like(positions, dtype=np.float64)

    for robot_i in range(n_robots):
        for robot_j in neighbors[robot_i]:
            command[robot_i] -= k * (positions[robot_i] - positions[robot_j])

    return command
