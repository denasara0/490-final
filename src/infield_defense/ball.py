from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ball_position(
    t: float,
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Where the ball is at time ``t`` after the bounce/roll started.

    Args:
        t: Time in seconds since the ball state (p0, v0, a) was fixed.
        p0: Starting position [x, y].
        v0: Starting velocity [vx, vy].
        a: Constant acceleration [ax, ay] (often negative along the direction of motion = slowing down).

    Returns:
        A new array [x, y] = p0 + v0*t + 0.5*a*t^2
    """
    t = float(t)
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    
    return p0 + v0 * t + 0.5 * a * (t**2)


def ball_velocity(
    t: float,
    v0: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(v0, dtype=np.float64) + np.asarray(a, dtype=np.float64) * float(t)
