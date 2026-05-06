# Multi-robot infield defense (MRS 490 final)

Distributed intercept and relay coordination inspired by baseball infield play: formation control, ground-ball kinematics, time-to-intercept bidding, and runner-aware force-out resolution around the bases (see project proposal).

## Environment

Python 3.10+ recommended (tested with 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the visualization scaffold

```bash
infield-sim
# or
python -m infield_defense.cli
```

This opens a matplotlib animation: robots hold a ring formation, then a sample ground ball is launched, a batter-runner circles the bases, and the defense must field the ball and beat the runner to the current force base.

## Report

- LaTeX report source lives at `report/infield_defense_paper.tex`.

## Run repeated-trial evaluation (quantitative metrics)

```bash
infield-eval
# or
python -m infield_defense.evaluate
```

This runs repeated trials across a small sweep of \(N\) and \(v_{\max}\), compares baselines, and writes `results/eval.csv` for tables/plots.
Each episode is labeled as `out` (success), `run` (failure), or `foul` (null), and the aggregate table reports counts/rates for all three outcomes.

## Analyze evaluation output

```bash
infield-analyze
# or
python -m infield_defense.analyze_eval
```

This reads `results/eval.csv`, refreshes `results/eval_summary.csv`, and writes plots plus helper CSVs under `results/analysis/`.

## Layout

- `src/infield_defense/config.py` — field bounds, `v_max`, formation gain, timestep
- `src/infield_defense/ball.py` — trajectory \(p_b(t) = p_0 + v_0 t + \frac{1}{2} a t^2\)
- `src/infield_defense/formation.py` — decentralized spacing control on a ring graph
- `src/infield_defense/coordination.py` — trajectory-aware fielding planner, interception bids, and delivery cost (Algorithms 1–2)
- `src/infield_defense/baserunning.py` — runner state and basepath progression helpers
- `src/infield_defense/simulation.py` — shared play state and timestep hooks for headless + live simulation
- `src/infield_defense/cli.py` — demo viewer
- `src/infield_defense/evaluate.py` — headless repeated-trial runner (CSV + aggregate table)

Dependencies are **NumPy** and **Matplotlib** only (no Webots); you can add a physics engine later if needed.
