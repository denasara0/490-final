"""
Infield defense simulation — multiple robots coordinate like baseball infielders.

What this package does
    1. Robots spread out in a formation (stay roughly evenly spaced).
    2. When a ground ball appears, each robot estimates its moving-ball intercept time.
    3. A trajectory-aware planner picks the "primary fielder" using intercept time plus lane ownership bias.
    4. A batter-runner circles the bases, and the defense records an out only by beating the runner to the current force base.

Start here:
    Run ``python -m infield_defense.cli`` (or the ``infield-sim`` command) to see a simple animation.

Code map:
    ``config`` — numbers we can tweak
    ``ball`` — where the ball is at time t
    ``formation`` — how robots nudge each other to hold a shape.
    ``coordination`` — who should field the ball; direct vs relay time estimates
    ``baserunning`` — runner state and basepath progression
    ``simulation`` — shared play state and per-step resolution
    ``cli`` — draws everything with Matplotlib
"""

__version__ = "0.1.0"
