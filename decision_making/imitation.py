from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from decision_making.gridworld import Action, State, StochasticGridworld


@dataclass(frozen=True)
class EpisodeStep:
    state: State
    action: Action
    reward: float
    next_state: State


@dataclass(frozen=True)
class Episode:
    steps: tuple[EpisodeStep, ...]
    reached_goal: bool

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)


@dataclass(frozen=True)
class DaggerMetric:
    iteration: int
    accuracy: float
    rollout_reward: float
    reached_goal: bool


@dataclass(frozen=True)
class DaggerResult:
    learned_policy: dict[State, Action | None]
    metrics: tuple[DaggerMetric, ...]
    aggregated_examples: int


def _validate_policy(
    environment: StochasticGridworld, policy: Mapping[State, Action | None]
) -> None:
    for state in environment.states:
        if state == environment.goal_state:
            continue
        if policy.get(state) not in environment.actions:
            raise ValueError(f"Policy has no valid action for state {state}.")


def simulate_episode(
    environment: StochasticGridworld,
    policy: Mapping[State, Action | None],
    rng: np.random.Generator,
    start_state: State | None = None,
    max_steps: int = 200,
) -> Episode:
    if max_steps <= 0:
        raise ValueError("The rollout horizon must be positive.")
    _validate_policy(environment, policy)
    state = environment.start_state if start_state is None else start_state
    if not environment.in_bounds(state):
        raise ValueError("The rollout start state must be inside the grid.")
    if state == environment.goal_state:
        return Episode(steps=(), reached_goal=True)

    steps: list[EpisodeStep] = []
    for _ in range(max_steps):
        action = policy[state]
        assert action is not None
        transition = environment.sample_transition(state, action, rng)
        steps.append(
            EpisodeStep(
                state=state,
                action=action,
                reward=transition.reward,
                next_state=transition.next_state,
            )
        )
        state = transition.next_state
        if transition.terminated:
            return Episode(steps=tuple(steps), reached_goal=True)
    return Episode(steps=tuple(steps), reached_goal=False)


def policy_accuracy(
    environment: StochasticGridworld,
    candidate: Mapping[State, Action | None],
    expert: Mapping[State, Action | None],
) -> float:
    evaluation_states = [
        state for state in environment.states if state != environment.goal_state
    ]
    if not evaluation_states:
        raise ValueError("Accuracy requires at least one non-terminal state.")
    matches = sum(
        candidate.get(state) == expert.get(state) for state in evaluation_states
    )
    return matches / len(evaluation_states)


def run_dagger(
    environment: StochasticGridworld,
    expert_policy: Mapping[State, Action | None],
    checkpoints: Sequence[int] = (5, 10, 20, 30, 40, 50),
    seed: int = 17,
    max_episode_steps: int = 200,
    tree_max_depth: int | None = None,
) -> DaggerResult:
    _validate_policy(environment, expert_policy)
    ordered_checkpoints = tuple(sorted(set(checkpoints)))
    if not ordered_checkpoints or ordered_checkpoints[0] <= 0:
        raise ValueError("DAgger checkpoints must be positive integers.")

    rng = np.random.default_rng(seed)
    non_terminal_states = [
        state for state in environment.states if state != environment.goal_state
    ]
    learned_policy: dict[State, Action | None] = {
        state: environment.actions[int(rng.integers(len(environment.actions)))]
        for state in non_terminal_states
    }
    learned_policy[environment.goal_state] = None
    training_states: list[State] = []
    training_labels: list[int] = []
    metrics: list[DaggerMetric] = []

    for iteration in range(1, ordered_checkpoints[-1] + 1):
        rollout = simulate_episode(
            environment,
            learned_policy,
            rng,
            max_steps=max_episode_steps,
        )
        for step in rollout.steps:
            expert_action = expert_policy[step.state]
            assert expert_action is not None
            training_states.append(step.state)
            training_labels.append(environment.actions.index(expert_action))

        classifier = DecisionTreeClassifier(
            random_state=seed,
            max_depth=tree_max_depth,
        )
        classifier.fit(np.asarray(training_states), np.asarray(training_labels))
        predictions = classifier.predict(np.asarray(non_terminal_states))
        for state, action_index in zip(non_terminal_states, predictions):
            learned_policy[state] = environment.actions[int(action_index)]

        if iteration in ordered_checkpoints:
            evaluation_rng = np.random.default_rng(seed + 10_000 + iteration)
            evaluation = simulate_episode(
                environment,
                learned_policy,
                evaluation_rng,
                max_steps=max_episode_steps,
            )
            metrics.append(
                DaggerMetric(
                    iteration=iteration,
                    accuracy=policy_accuracy(
                        environment, learned_policy, expert_policy
                    ),
                    rollout_reward=evaluation.total_reward,
                    reached_goal=evaluation.reached_goal,
                )
            )

    return DaggerResult(
        learned_policy=learned_policy,
        metrics=tuple(metrics),
        aggregated_examples=len(training_states),
    )
