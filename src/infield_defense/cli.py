from __future__ import annotations

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Polygon, Wedge
from matplotlib.widgets import Button

from infield_defense.config import DEFAULT_DYNAMICS, DEFAULT_FIELD
from infield_defense.coordination import select_primary_fielder
from infield_defense.simulation import (
    clip_positions_to_infield,
    current_ball_pos,
    launch_ground_ball,
    make_state,
)


def main() -> None:
    dyn = DEFAULT_DYNAMICS
    field = DEFAULT_FIELD
    state = make_state(field=field)
    bases = {
        name: np.array(position, dtype=np.float64)
        for name, position in field.base_positions().items()
    }
    diamond_order = ["home", "first", "second", "third", "home"]
    diamond_xy = np.array([bases[name] for name in diamond_order], dtype=np.float64)
    rng = np.random.default_rng()
    hit_speed = 18.0
    fair_angle_min = 45.0
    fair_angle_max = 135.0
    catch_radius = 1.5
    foul_detection_radius = 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.18)
    ax.set_aspect("equal")
    ax.set_xlim(field.x_min, field.x_max)
    ax.set_ylim(field.y_min, field.y_max)
    ax.set_title("Multi-Robot Infield Defense System")
    ax.set_facecolor("#d8f0c0")
    ax.grid(False)

    home = bases["home"]
    first = bases["first"]
    second = bases["second"]
    third = bases["third"]
    outfield_radius = 95.0
    infield_radius = 55.0
    foul_right = home + outfield_radius * np.array([np.cos(np.deg2rad(45.0)), np.sin(np.deg2rad(45.0))])
    foul_left = home + outfield_radius * np.array([np.cos(np.deg2rad(135.0)), np.sin(np.deg2rad(135.0))])
    ax.add_patch(Wedge(home, outfield_radius, 45, 135, facecolor="#9dd38b", edgecolor="none", alpha=0.95))
    ax.add_patch(Wedge(home, infield_radius, 45, 135, facecolor="#cfa97a", edgecolor="none", alpha=0.7))
    ax.add_patch(Polygon([home, first, second, third], closed=True, facecolor="#dcb789", edgecolor="none", alpha=0.95))
    ax.add_patch(Arc(home, 2 * outfield_radius, 2 * outfield_radius, theta1=45, theta2=135, lw=2.0, color="#2f6d3b"))
    ax.plot([home[0], foul_right[0]], [home[1], foul_right[1]], color="white", lw=2.5)
    ax.plot([home[0], foul_left[0]], [home[1], foul_left[1]], color="white", lw=2.5)

    (robot_line,) = ax.plot([], [], "o", ms=10, label="infielders")
    (diamond_line,) = ax.plot(
        diamond_xy[:, 0],
        diamond_xy[:, 1],
        color="white",
        lw=2.5,
        label="base paths",
    )
    base_xy = np.array(list(bases.values()), dtype=np.float64)
    (base_pt,) = ax.plot(base_xy[:, 0], base_xy[:, 1], "s", ms=11, color="white", mec="saddlebrown", label="bases")
    (ball_pt,) = ax.plot([], [], "o", ms=8, color="red", label="ball")
    role_texts = [
        ax.text(0.0, 0.0, role_name, fontsize=9, ha="center", va="bottom", color="#1d3557")
        for role_name in state.role_names
    ]
    for name, xy in bases.items():
        ax.text(xy[0] + 1.2, xy[1] + 1.2, name.title(), fontsize=9, color="saddlebrown")
    status_text = ax.text(
        0.02,
        0.98,
        "Waiting for pitch",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    scoreboard = {"outs": 0, "fouls": 0, "home_runs": 0}
    scoreboard_text = ax.text(
        0.02,
        0.89,
        "Outs: 0  Fouls: 0  Home runs: 0",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.text(
        0.5,
        0.04,
        "Space/Enter: hit   R: reset play",
        ha="center",
        va="center",
    )
    ax.legend(loc="upper right")

    simulation_time = 0.0
    ball_has_been_launched = False
    ball_display_xy: np.ndarray | None = None
    current_hit_is_fair = True

    button_width = 0.22
    button_height = 0.06
    button_gap = 0.04
    total_button_width = 2 * button_width + button_gap
    left_edge = 0.5 - total_button_width / 2
    pitch_ax = fig.add_axes([left_edge, 0.08, button_width, button_height])
    reset_ax = fig.add_axes([left_edge + button_width + button_gap, 0.08, button_width, button_height])
    hit_button = Button(pitch_ax, "Hit Ball")
    reset_button = Button(reset_ax, "Reset Play")

    def update_role_labels() -> None:
        for index, text in enumerate(role_texts):
            text.set_position((state.positions[index, 0], state.positions[index, 1] + 2.2))

    def update_scoreboard() -> None:
        scoreboard_text.set_text(
            f"Outs: {scoreboard['outs']}  Fouls: {scoreboard['fouls']}  Home runs: {scoreboard['home_runs']}"
        )

    def distance_from_home(ball_xy: np.ndarray) -> float:
        return float(np.linalg.norm(ball_xy - home))

    def is_in_fair_territory(ball_xy: np.ndarray) -> bool:
        relative_xy = np.asarray(ball_xy, dtype=np.float64) - home
        return float(relative_xy[1]) >= abs(float(relative_xy[0]))

    def stop_play(result_text: str, ball_xy: np.ndarray) -> None:
        nonlocal ball_display_xy
        ball_display_xy = np.asarray(ball_xy, dtype=np.float64).copy()
        state.ball.active = False
        state.velocities[:] = 0.0
        status_text.set_text(result_text)
        fig.canvas.draw_idle()

    def launch_sample_ball(_event=None) -> None:
        nonlocal ball_has_been_launched, ball_display_xy, current_hit_is_fair
        if ball_has_been_launched:
            status_text.set_text("Ball already in play. Press Reset to pitch again.")
            fig.canvas.draw_idle()
            return

        angle_deg = float(rng.uniform(20.0, 160.0))
        angle_rad = np.deg2rad(angle_deg)
        ball_velocity = hit_speed * np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)
        launch_ground_ball(
            state,
            simulation_time,
            p0=home,
            v0=ball_velocity,
            decel_magnitude=0.0,
        )
        ball_display_xy = home.copy()
        ball_has_been_launched = True
        current_hit_is_fair = fair_angle_min <= angle_deg <= fair_angle_max
        trajectory_label = "fair" if current_hit_is_fair else "foul"
        status_text.set_text(f"Ball in play: {trajectory_label} trajectory at {angle_deg:.0f} degrees")
        fig.canvas.draw_idle()

    def reset_play(_event=None) -> None:
        nonlocal state, simulation_time, ball_has_been_launched, ball_display_xy, current_hit_is_fair
        state = make_state(field=field)
        simulation_time = 0.0
        ball_has_been_launched = False
        ball_display_xy = None
        current_hit_is_fair = True
        robot_line.set_data(state.positions[:, 0], state.positions[:, 1])
        ball_pt.set_data([], [])
        status_text.set_text("Waiting for pitch")
        update_role_labels()
        fig.canvas.draw_idle()

    def on_key_press(event) -> None:
        if event.key in {" ", "space", "enter"}:
            launch_sample_ball()
        elif event.key in {"r", "R"}:
            reset_play()

    hit_button.on_clicked(launch_sample_ball)
    reset_button.on_clicked(reset_play)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    def init():
        robot_line.set_data(state.positions[:, 0], state.positions[:, 1])
        ball_pt.set_data([], [])
        update_role_labels()
        update_scoreboard()
        return robot_line, diamond_line, base_pt, ball_pt, status_text, scoreboard_text, *role_texts

    def update(_frame: int):
        nonlocal simulation_time, ball_display_xy
        simulation_time += dyn.dt

        if state.ball.active and state.primary_idx is not None:
            ball_xy = current_ball_pos(state, simulation_time)
            ball_display_xy = ball_xy
            state.primary_idx = select_primary_fielder(state.positions, ball_xy, dyn.vmax)
            for robot_index in range(state.positions.shape[0]):
                target = ball_xy if robot_index == state.primary_idx else state.home_positions[robot_index]
                vector_to_target = target - state.positions[robot_index]
                distance_to_target = float(np.linalg.norm(vector_to_target))
                if distance_to_target > 0.1:
                    direction = vector_to_target / distance_to_target
                    state.velocities[robot_index] = direction * dyn.vmax
                else:
                    state.velocities[robot_index] = np.zeros(2)
            state.positions += state.velocities * dyn.dt
            clip_positions_to_infield(state.positions, field)

            foul_crossed = (
                distance_from_home(ball_xy) >= foul_detection_radius
                and not is_in_fair_territory(ball_xy)
            )
            if foul_crossed:
                scoreboard["fouls"] += 1
                update_scoreboard()
                stop_play("Foul ball", ball_xy)
            distances_to_ball = np.linalg.norm(state.positions - ball_xy, axis=1)
            fielder_idx = int(np.argmin(distances_to_ball))
            if state.ball.active and float(distances_to_ball[fielder_idx]) <= catch_radius:
                scoreboard["outs"] += 1
                update_scoreboard()
                stop_play(f"Out recorded by {state.role_names[fielder_idx]}", ball_xy)
            elif state.ball.active and distance_from_home(ball_xy) >= outfield_radius:
                scoreboard["home_runs"] += 1
                update_scoreboard()
                stop_play("Home run", ball_xy)
        else:
            state.velocities[:] = 0.0

        robot_line.set_data(state.positions[:, 0], state.positions[:, 1])
        update_role_labels()
        if ball_display_xy is not None:
            ball_pt.set_data([ball_display_xy[0]], [ball_display_xy[1]])
        else:
            ball_pt.set_data([], [])
        return robot_line, diamond_line, base_pt, ball_pt, status_text, scoreboard_text, *role_texts

    _anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=int(dyn.dt * 1000),
        blit=False,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
