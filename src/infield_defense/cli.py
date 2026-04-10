"""
Run a simple animated window using Matplotlib.

Beginner notes:
    - ``main`` builds the figure once, then ``FuncAnimation`` calls ``update`` many times per second.
    - ``nonlocal`` lets ``update`` change variables that belong to ``main`` (simulation time and a flag).
    - The animation object is stored in ``_anim`` so it is not garbage-collected before the window opens.
"""

from __future__ import annotations

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD
from infield_defense.formation import formation_control_input
from infield_defense.simulation import (
    current_ball_pos,
    launch_ground_ball,
    make_state,
    step_formation,
)


def neighbor_indices_sub(indices: list[int], full_neighbors: list[list[int]]) -> list[list[int]]:
    """
    When only some robots follow formation rules, rebuild a smaller neighbor list.

    ``indices`` are the original robot numbers still in "formation mode"; we map them to 0..k-1
    so ``formation_control_input`` can run on just that subset.
    """
    original_to_subset = {original: subset_i for subset_i, original in enumerate(indices)}
    sub_neighbors: list[list[int]] = []

    for original_i in indices:
        subset_neighbors: list[int] = []
        for original_j in full_neighbors[original_i]:
            if original_j in original_to_subset:
                subset_neighbors.append(original_to_subset[original_j])
        # If someone had no neighbor in the subset, fall back to index 0 to avoid an empty sum.
        sub_neighbors.append(subset_neighbors if subset_neighbors else [0])

    return sub_neighbors


def main() -> None:
    """Open a window: robots form a ring, then a sample grounder is hit."""
    dyn = DEFAULT_DYNAMICS
    field = DEFAULT_FIELD
    state = make_state(n_robots=5)
    base = np.array(field.first_base, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(field.x_min, field.x_max)
    ax.set_ylim(field.y_min, field.y_max)
    ax.set_title("Infield defense demo — formation, then chase (starter project)")
    ax.grid(True, alpha=0.3)

    (robot_line,) = ax.plot([], [], "o", ms=10, label="robots")
    (base_pt,) = ax.plot([base[0]], [base[1]], "s", ms=12, color="tan", label="first base")
    (ball_pt,) = ax.plot([], [], "o", ms=8, color="red", label="ball")
    ax.legend(loc="upper right")

    # These live in main() but get updated inside update(); nonlocal is required for that.
    simulation_time = 0.0
    ball_has_been_launched = False

    def init():
        robot_line.set_data(state.positions[:, 0], state.positions[:, 1])
        ball_pt.set_data([], [])
        return robot_line, base_pt, ball_pt

    def update(_frame: int):
        nonlocal simulation_time, ball_has_been_launched
        simulation_time += dyn.dt

        # After half a second, create one sample ground ball (easy to find in the code to edit).
        if not ball_has_been_launched and simulation_time > 0.5:
            launch_ground_ball(
                state,
                simulation_time,
                p0=np.array([-30.0, -5.0], dtype=np.float64),
                v0=np.array([14.0, 2.0], dtype=np.float64),
                decel_magnitude=dyn.ball_decel,
            )
            ball_has_been_launched = True

        if state.ball.active and state.primary_idx is not None:
            ball_xy = current_ball_pos(state, simulation_time)

            # Chosen fielder runs toward the current ball position (straight line, max speed).
            primary = state.primary_idx
            vector_to_ball = ball_xy - state.positions[primary]
            distance_to_ball = float(np.linalg.norm(vector_to_ball))
            close_enough = 0.1
            if distance_to_ball > close_enough:
                direction = vector_to_ball / distance_to_ball
                state.velocities[primary] = direction * dyn.vmax
            else:
                state.velocities[primary] = np.zeros(2)

            # Everyone else keeps using the formation rule on the remaining robots.
            non_primary = [i for i in range(state.positions.shape[0]) if i != primary]
            if non_primary:
                positions_subset = state.positions[non_primary]
                neighbor_subset = neighbor_indices_sub(non_primary, state.neighbors)
                raw = formation_control_input(positions_subset, neighbor_subset, dyn.formation_gain)
                for k, robot_index in enumerate(non_primary):
                    state.velocities[robot_index] = raw[k]
                for robot_index in non_primary:
                    v = state.velocities[robot_index]
                    speed = float(np.linalg.norm(v))
                    if speed > dyn.vmax:
                        state.velocities[robot_index] = v * (dyn.vmax / speed)

            state.positions += state.velocities * dyn.dt
        else:
            step_formation(state, field, dyn)

        robot_line.set_data(state.positions[:, 0], state.positions[:, 1])
        if state.ball.active:
            ball_xy = current_ball_pos(state, simulation_time)
            ball_pt.set_data([ball_xy[0]], [ball_xy[1]])
        return robot_line, base_pt, ball_pt

    # Keep a reference; otherwise Python might delete the animation while the window is open.
    _anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=int(dyn.dt * 1000),
        blit=True,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
