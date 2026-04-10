"""
Infield defense simulation — multiple robots coordinate like baseball infielders.

What this package does 
    1. Robots spread out in a formation (stay roughly evenly spaced).
    2. When a ground ball appears, each robot estimates how long it would take to reach the ball.
    3. The robot with the shortest time becomes the "primary fielder" and goes for the ball.
    4. Later you can add: pass the ball to a teammate (relay) or run straight to base (direct).

Start here:
    Run ``python -m infield_defense.cli`` (or the ``infield-sim`` command) to see a simple animation.

Code map:
    ``config`` — numbers we can tweak 
    ``ball`` — where the ball is at time t 
    ``formation`` — how robots nudge each other to hold a shape.
    ``coordination`` — who should field the ball; direct vs relay time estimates
    ``simulation`` — stores positions and advances time one small step
    ``cli`` — draws everything with Matplotlib
"""

__version__ = "0.1.0"
