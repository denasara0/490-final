

## Proposal alignment

| Proposal topic | Status | Notes |
|----------------|--------|--------|
| §II System model (2D plane, kinematic robots) | **Done** | `config.py`, `simulation.py` |
| §III Ball trajectory (eq. 5) | **Done** | `ball.py` |
| §IV Formation control (eq. 6) | **Done** | `formation.py` (ring neighbors) |
| §V–VI Interception time & allocation | **Partial** | Estimates + argmin in `coordination.py`; chase uses **current** ball position in demo, not full trajectory intercept |
| §VII Algorithm 2 (delivery optimization) | **Partial** | `delivery_costs()` implemented; **not** yet driving robot behavior or animation |
| §VIII Simulation & visualization | **Partial** | Live 2D plot; no batch/headless runner yet |
| §VIII Experiments / metrics | **Not started** | Retrieval time, formation stability, connectivity, robustness — to be logged or computed |

---

## Completed work

1. **Repository and environment**  
   - `pyproject.toml`, editable install, `requirements.txt`, `.gitignore`, optional dev deps (pytest, ruff).

2. **Core simulation modules** (`src/infield_defense/`)  
   - **Config:** field bounds, first-base target, `v_max`, formation gain, timestep, ball deceleration.  
   - **Ball:** constant-acceleration trajectory in 2D.  
   - **Formation:** ring neighbor graph and formation control input.  
   - **Coordination:** interception time, primary fielder index, direct vs relay time comparison.  
   - **Simulation:** world state, Euler steps, formation stepping, ground-ball launch hook, primary assignment on launch.

3. **Visualization**  
   - `cli.py`: animated field, robots, base, ball; formation then sample grounder and primary chase.

4. **Documentation for onboarding**  
   - Module docstrings and comments aimed at beginners; `README.md` describes setup and module map.

---

## In progress / limitations

- **Intercept behavior:** Demo moves the primary toward the **instantaneous** ball location; the proposal’s use of **predicted** positions along the trajectory can be strengthened (e.g., minimize time-to-intercept along the parabola/roll).  
- **Distributed aspect:** Each robot *could* compute its own bid; the codebase computes times centrally for simplicity. A message-passing narrative or explicit broadcast simulation may be added for the final report if needed.  
- **End-to-end play:** No scripted sequence yet for *secure ball → choose direct/relay → execute pass/run → reset formation*.  
- **Evaluation:** No automated logging, plots, or sweeps over speeds / formations.

---

## Planned next steps

1. Wire **`delivery_costs`** into the simulation after the ball is “secured” (distance threshold), then animate direct run to base or relay handoff.  
2. Improve **interception** using predicted ball trajectory (discrete search or closed-form where feasible).  
3. Add **metrics** module: time to intercept, time to delivery, simple connectivity checks on the ring graph, optional noise/failure trials for robustness.  
4. Add **headless** mode (no GUI) for repeatable runs and CSV/plot outputs.  
5. Add **tests** for `ball_position`, `formation_control_input`, `select_primary_fielder`, and `delivery_costs`.  
6. Draft **final report / video** using recorded runs from the visualization or exported plots.

---

## Blockers and risks

- **Model fidelity:** Straight-line time estimates are easy to explain but optimistic; results should be framed as lower bounds unless obstacle avoidance is added.  
- **Scope:** Adding a full physics engine (proposal considered Python physics libraries) would improve realism but increases integration time; current stack stays lightweight (NumPy + Matplotlib).

---

## Summary status

**Overall:** Foundation and several proposal components are **implemented and runnable**; **integration of relay play, trajectory-based intercept, and experimental metrics** are the main gaps before the project fully reflects the written algorithms and evaluation plan.

---

*This file should be updated as milestones close*
