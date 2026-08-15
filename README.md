# Planning and Imitation Learning

A reproducible implementation of sequential decision-making agents, from exact planning in a stochastic Markov decision process to policy imitation with Dataset Aggregation (DAgger).

## What I learned

- Modeling states, actions, transition probabilities, rewards, terminal states, and the Markov property
- Designing safe reward functions and reasoning about discounting and Bellman contraction
- Solving finite MDPs with value iteration and policy iteration
- Evaluating fixed policies with iterative Bellman backups and linear systems
- Representing stochastic action outcomes and consolidating duplicate transition probability mass
- Simulating finite-horizon policies with deterministic random-number control
- Training a decision-tree policy with DAgger to address covariate shift in imitation learning
- Comparing policies only on actionable, non-terminal states
- Building repeatable experiments with explicit convergence tolerances and failure bounds

## Project structure

| Path | Purpose |
| --- | --- |
| `decision_making/gridworld.py` | Validated stochastic gridworld dynamics and transition sampling |
| `decision_making/planning.py` | Value iteration, policy evaluation, policy iteration, and result formatting |
| `decision_making/imitation.py` | Bounded policy rollouts, policy agreement, and DAgger |
| `run_experiments.py` | Reproduces the planning comparison and imitation-learning curve |
| `tests/` | Dynamics, convergence, numerical-equivalence, reproducibility, and regression tests |
| `reports/` | Original written analyses of MDPs, planning, and imitation learning |

The default environment is a 4-by-4 grid with stochastic movement, water and wildfire penalties, and a terminal goal reward. An intended action succeeds with probability 0.8 and slips to either perpendicular direction with probability 0.1.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py
```

The experiment compares discount factors 0.3 and 0.95, verifies that value iteration and policy iteration converge to the same optimum, runs a seeded expert rollout, and evaluates DAgger at 5, 10, 20, 30, 40, and 50 aggregation iterations. The learning curve is written to the ignored `artifacts/` directory. Use `python run_experiments.py --help` to change the seed, checkpoints, rollout horizon, or plot location.
