"""
All the main "knobs" for the simulation in one place.

If you are new to the project, try changing:
    - ``DynamicsConfig.vmax`` — how fast robots can move
    - ``DynamicsConfig.dt`` — smaller = smoother but slower to run
    - ``FieldConfig`` limits — how big the playing area is

Units are abstract "meters" on a 2D plane; what matters is ratios, not real-world exactness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldConfig:
    """
    The rectangle where robots are allowed to move, plus where "first base" is.

    Attributes:
        x_min, x_max: Left and right edges of the field.
        y_min, y_max: Bottom and top edges of the field.
        first_base: (x, y) point the team wants to deliver the ball toward.
    """

    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = -40.0
    y_max: float = 40.0
    first_base: tuple[float, float] = (35.0, 0.0)


@dataclass(frozen=True)
class DynamicsConfig:
    """
    How fast things move and how often we update the simulation.

    Attributes:
        vmax: Maximum robot speed (distance per second in our abstract units).
        formation_gain: How strongly robots react to neighbors (higher = snappier spacing).
        dt: Length of one simulation step in seconds (used in position += velocity * dt).
        ball_decel: How much the ground slows the ball horizontally (friction-style effect).
    """

    vmax: float = 6.0
    formation_gain: float = 0.8
    dt: float = 0.05
    ball_decel: float = 2.5


# Ready-made defaults so other modules can import one object instead of building their own.
DEFAULT_FIELD = FieldConfig()
DEFAULT_DYNAMICS = DynamicsConfig()
