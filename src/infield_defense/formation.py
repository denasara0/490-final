"""
Formation control: robots stay spread out using only local information.

Idea (from the proposal):
    Each robot looks at its *neighbors* (robots it can "talk" to).
    If it is too far from a neighbor, it gets pulled closer; if the math pushes the other way,
    neighbors balance out. On a ring of robots, this tends to make spacing even.

The control rule for robot i is::

    u_i = -k * sum over neighbors j of (position_i - position_j)

``u_i`` acts like a desired velocity before we cap speed — see ``simulation.clip_speed``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def neighbor_indices_ring(n: int) -> list[list[int]]:
    """
    Build a "who talks to whom" list for robots arranged in a ring.

    Robot i is neighbors with i-1 and i+1 (wrapping around), like people holding hands in a circle.

    Args:
        n: How many robots.

    Returns:
        A list of length n; entry i is [left_neighbor_index, right_neighbor_index].
    """
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
    """
    Compute a velocity-like command for each robot to maintain formation.

    Args:
        positions: Shape (N, 2) — row i is robot i's [x, y].
        neighbors: neighbors[i] lists robot indices that robot i uses in the sum.
        k: Positive gain; larger values make corrections stronger (can overshoot if too large).

    Returns:
        Same shape as positions; row i is the raw command u_i before speed limiting.
    """
    n_robots = positions.shape[0]
    command = np.zeros_like(positions, dtype=np.float64)

    for robot_i in range(n_robots):
        for robot_j in neighbors[robot_i]:
            # (p_i - p_j) measures how robot i is shifted relative to j; summing pulls the team together.
            command[robot_i] -= k * (positions[robot_i] - positions[robot_j])

    return command
