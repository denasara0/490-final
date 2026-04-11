from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldConfig:
    x_min: float = -75.0
    x_max: float = 75.0
    y_min: float = -42.0
    y_max: float = 68.0
    home_base: tuple[float, float] = (0.0, -35.0)
    first_base: tuple[float, float] = (35.0, 0.0)
    second_base: tuple[float, float] = (0.0, 35.0)
    third_base: tuple[float, float] = (-35.0, 0.0)
    infield_x_limit: float = 28.0
    infield_y_min: float = -8.0
    infield_y_max: float = 28.0

    def base_positions(self) -> dict[str, tuple[float, float]]:
        return {
            "home": self.home_base,
            "first": self.first_base,
            "second": self.second_base,
            "third": self.third_base,
        }

    def infielder_positions(self) -> dict[str, tuple[float, float]]:
        return {
            "Shortstop": (0,0),
            "1st baseman": (35,0),
            "2nd baseman": (0,35),
            "3rd baseman": (-35,0),
        }


@dataclass(frozen=True)
class DynamicsConfig:
    vmax: float = 6.0
    formation_gain: float = 0.8
    dt: float = 0.05
    ball_decel: float = 2.5


DEFAULT_FIELD = FieldConfig()
DEFAULT_DYNAMICS = DynamicsConfig()
