from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from infield_defense.config import DEFAULT_FIELD, FieldConfig


@dataclass
class RunnerState:
    position: npt.NDArray[np.float64]
    next_base_index: int = 1
    active: bool = False
    out: bool = False
    scored: bool = False


def make_runner(field: FieldConfig = DEFAULT_FIELD) -> RunnerState:
    home = np.asarray(field.base_positions()["home"], dtype=np.float64)
    return RunnerState(position=home.copy())


def reset_runner(runner: RunnerState, field: FieldConfig = DEFAULT_FIELD) -> None:
    home = np.asarray(field.base_positions()["home"], dtype=np.float64)
    runner.position = home.copy()
    runner.next_base_index = 1
    runner.active = False
    runner.out = False
    runner.scored = False


def start_runner(runner: RunnerState, field: FieldConfig = DEFAULT_FIELD) -> None:
    reset_runner(runner, field=field)
    runner.active = True


def current_target_base_name(
    runner: RunnerState,
    field: FieldConfig = DEFAULT_FIELD,
) -> str | None:
    if not runner.active:
        return None
    base_order = field.base_path_order()
    if runner.next_base_index >= len(base_order):
        return None
    return base_order[runner.next_base_index]


def current_target_base_position(
    runner: RunnerState,
    field: FieldConfig = DEFAULT_FIELD,
) -> npt.NDArray[np.float64] | None:
    base_name = current_target_base_name(runner, field=field)
    if base_name is None:
        return None
    return np.asarray(field.base_positions()[base_name], dtype=np.float64)


def mark_runner_out(runner: RunnerState) -> None:
    runner.active = False
    runner.out = True
    runner.scored = False


def advance_runner(
    runner: RunnerState,
    *,
    dt: float,
    speed: float,
    field: FieldConfig = DEFAULT_FIELD,
    touch_radius: float = 0.75,
) -> str | None:
    if not runner.active:
        return None

    target_name = current_target_base_name(runner, field=field)
    target_pos = current_target_base_position(runner, field=field)
    if target_name is None or target_pos is None:
        runner.active = False
        return None

    delta = target_pos - runner.position
    distance = float(np.linalg.norm(delta))
    step_distance = max(0.0, float(speed)) * max(0.0, float(dt))

    if distance <= max(0.0, float(touch_radius)) or step_distance >= distance:
        runner.position = target_pos.copy()
    elif distance > 1e-12:
        runner.position = runner.position + (delta / distance) * step_distance

    remaining = float(np.linalg.norm(target_pos - runner.position))
    if remaining > max(0.0, float(touch_radius)):
        return None

    runner.next_base_index += 1
    if runner.next_base_index >= len(field.base_path_order()):
        runner.active = False
        runner.scored = True
        return "home"
    return target_name
