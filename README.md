# Reinforcement Learning and Decision Systems

A reproducible collection of sequential decision-making experiments, spanning exact planning, online reinforcement learning, actor-critic methods, and policy imitation.

## What I learned

- Modeling states, actions, transition probabilities, rewards, terminal states, and the Markov property
- Designing safe reward functions and reasoning about discounting and Bellman contraction
- Solving finite MDPs with value iteration and policy iteration
- Evaluating fixed policies with iterative Bellman backups and linear systems
- Learning action values online with SARSA and Q-learning
- Assigning credit across multiple steps with backward eligibility traces in SARSA(lambda)
- Combining a softmax policy-gradient actor with a linear TD critic
- Comparing on-policy, off-policy, and bootstrapped control methods across repeated trials
- Reasoning about first-visit and every-visit Monte Carlo estimation and potential-based reward shaping
- Representing stochastic action outcomes and consolidating duplicate transition probability mass
- Simulating finite-horizon policies with deterministic random-number control
- Training a decision-tree policy with DAgger to address covariate shift in imitation learning
- Comparing policies only on actionable, non-terminal states
- Building repeatable experiments with explicit convergence tolerances and failure bounds
- Evaluating research on efficient exploration, learned world models, multi-robot coordination, reinforcement learning from human feedback, and maximum-entropy control

## Project structure

| Path | Purpose |
| --- | --- |
| `decision_making/gridworld.py` | Validated stochastic gridworld dynamics and transition sampling |
| `decision_making/planning.py` | Value iteration, policy evaluation, policy iteration, and result formatting |
| `decision_making/imitation.py` | Bounded policy rollouts, policy agreement, and DAgger |
| `decision_making/temporal_difference.py` | SARSA, Q-learning, eligibility traces, and actor-critic training |
| `run_experiments.py` | Reproduces the planning comparison and imitation-learning curve |
| `run_learning_experiments.py` | Repeats and compares the online reinforcement-learning experiments |
| `tests/` | Dynamics, convergence, numerical-equivalence, reproducibility, and regression tests |
| `reports/` | Original written analyses, experimental results, and research review |

The default environment is a 4-by-4 grid with stochastic movement, water and wildfire penalties, and a terminal goal reward. An intended action succeeds with probability 0.8 and slips to either perpendicular direction with probability 0.1.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiments.py
python run_learning_experiments.py
```

The planning experiment compares discount factors 0.3 and 0.95, verifies that value iteration and policy iteration converge to the same optimum, runs a seeded expert rollout, and evaluates DAgger at multiple aggregation checkpoints. The online-learning experiment compares SARSA, Q-learning, SARSA(lambda), and actor-critic over repeated seeded trials, summarizing performance over a configurable trailing episode window instead of a single noisy endpoint. Generated learning curves are written to the ignored `artifacts/` directory. Both commands expose their seeds, rollout horizons, and plot locations through `--help`.
