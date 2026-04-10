# Multi-robot infield defense (MRS 490 final)

Distributed intercept and relay coordination inspired by baseball infield play: formation control, ground-ball kinematics, time-to-intercept bidding, and direct vs relay delivery to base (see project proposal).

## Environment

Python 3.10+ recommended (tested with 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the visualization scaffold

```bash
infield-sim
# or
python -m infield_defense.cli
```

This opens a matplotlib animation: robots hold a ring formation, then a sample ground ball is launched and a primary fielder is chosen by lowest estimated interception time.

## Layout

- `src/infield_defense/config.py` — field bounds, `v_max`, formation gain, timestep
- `src/infield_defense/ball.py` — trajectory \(p_b(t) = p_0 + v_0 t + \frac{1}{2} a t^2\)
- `src/infield_defense/formation.py` — decentralized spacing control on a ring graph
- `src/infield_defense/coordination.py` — interception bids and delivery cost (Algorithms 1–2)
- `src/infield_defense/simulation.py` — state and timestep hooks
- `src/infield_defense/cli.py` — demo viewer

Dependencies are **NumPy** and **Matplotlib** only (no Webots); you can add a physics engine later if needed.
