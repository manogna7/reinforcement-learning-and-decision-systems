from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from decision_making.gridworld import Action, State, StochasticGridworld

ValueFunction: TypeAlias = dict[State, float]
Policy: TypeAlias = dict[State, Action | None]


@dataclass(frozen=True)
class PlanningResult:
    values: ValueFunction
    policy: Policy
    iterations: int


@dataclass(frozen=True)
class EvaluationResult:
    values: ValueFunction
    iterations: int


def _validate_discount(discount: float) -> None:
    if not 0.0 <= discount < 1.0:
        raise ValueError("The discount factor must satisfy 0 <= discount < 1.")


def _validate_solver_settings(tolerance: float, max_iterations: int) -> None:
    if tolerance <= 0:
        raise ValueError("The convergence tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("The iteration limit must be positive.")


def _validate_policy(
    environment: StochasticGridworld, policy: Mapping[State, Action | None]
) -> None:
    for state in environment.states:
        action = policy.get(state)
        if state == environment.goal_state:
            if action is not None:
                raise ValueError("The terminal state policy must be None.")
        elif action not in environment.actions:
            raise ValueError(f"Policy has no valid action for state {state}.")


def action_value(
    environment: StochasticGridworld,
    state: State,
    action: Action,
    values: Mapping[State, float],
    discount: float,
) -> float:
    return sum(
        transition.probability
        * (transition.reward + discount * values[transition.next_state])
        for transition in environment.transitions(state, action)
    )


def _greedy_action(
    environment: StochasticGridworld,
    state: State,
    values: Mapping[State, float],
    discount: float,
    preferred_action: Action | None = None,
    tolerance: float = 1e-12,
) -> Action:
    candidates = [
        (
            action,
            action_value(environment, state, action, values, discount),
        )
        for action in environment.actions
    ]
    best_value = max(value for _, value in candidates)
    tied_actions = [
        action
        for action, value in candidates
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=tolerance)
    ]
    if preferred_action in tied_actions:
        return preferred_action
    return tied_actions[0]


def _greedy_policy(
    environment: StochasticGridworld,
    values: Mapping[State, float],
    discount: float,
) -> Policy:
    return {
        state: (
            None
            if state == environment.goal_state
            else _greedy_action(environment, state, values, discount)
        )
        for state in environment.states
    }


def value_iteration(
    environment: StochasticGridworld,
    discount: float = 0.95,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> PlanningResult:
    _validate_discount(discount)
    _validate_solver_settings(tolerance, max_iterations)
    values = {state: 0.0 for state in environment.states}

    for iteration in range(1, max_iterations + 1):
        updated = values.copy()
        for state in environment.states:
            if state == environment.goal_state:
                updated[state] = 0.0
                continue
            updated[state] = max(
                action_value(environment, state, action, values, discount)
                for action in environment.actions
            )
        residual = max(abs(updated[state] - values[state]) for state in values)
        values = updated
        if residual < tolerance:
            return PlanningResult(
                values=values,
                policy=_greedy_policy(environment, values, discount),
                iterations=iteration,
            )
    raise RuntimeError("Value iteration did not converge within the iteration limit.")


def policy_evaluation_iterative(
    environment: StochasticGridworld,
    policy: Mapping[State, Action | None],
    discount: float = 0.95,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> EvaluationResult:
    _validate_discount(discount)
    _validate_solver_settings(tolerance, max_iterations)
    _validate_policy(environment, policy)
    values = {state: 0.0 for state in environment.states}

    for iteration in range(1, max_iterations + 1):
        updated = values.copy()
        for state in environment.states:
            if state == environment.goal_state:
                updated[state] = 0.0
                continue
            action = policy[state]
            assert action is not None
            updated[state] = action_value(environment, state, action, values, discount)
        residual = max(abs(updated[state] - values[state]) for state in values)
        values = updated
        if residual < tolerance:
            return EvaluationResult(values=values, iterations=iteration)
    raise RuntimeError("Policy evaluation did not converge within the iteration limit.")


def policy_evaluation_exact(
    environment: StochasticGridworld,
    policy: Mapping[State, Action | None],
    discount: float = 0.95,
) -> ValueFunction:
    _validate_discount(discount)
    _validate_policy(environment, policy)
    states = environment.states
    state_index = {state: index for index, state in enumerate(states)}
    transition_matrix = np.zeros((len(states), len(states)), dtype=float)
    reward_vector = np.zeros(len(states), dtype=float)

    for state in states:
        row = state_index[state]
        if state == environment.goal_state:
            continue
        action = policy[state]
        assert action is not None
        for transition in environment.transitions(state, action):
            column = state_index[transition.next_state]
            transition_matrix[row, column] += transition.probability
            reward_vector[row] += transition.probability * transition.reward

    system = np.eye(len(states)) - discount * transition_matrix
    solution = np.linalg.solve(system, reward_vector)
    return {state: float(solution[state_index[state]]) for state in states}


def policy_iteration(
    environment: StochasticGridworld,
    discount: float = 0.95,
    max_iterations: int = 1_000,
) -> PlanningResult:
    _validate_discount(discount)
    if max_iterations <= 0:
        raise ValueError("The iteration limit must be positive.")
    policy: Policy = {
        state: None if state == environment.goal_state else environment.actions[0]
        for state in environment.states
    }

    for iteration in range(1, max_iterations + 1):
        values = policy_evaluation_exact(environment, policy, discount)
        improved = policy.copy()
        stable = True
        for state in environment.states:
            if state == environment.goal_state:
                continue
            current_action = policy[state]
            assert current_action is not None
            improved_action = _greedy_action(
                environment,
                state,
                values,
                discount,
                preferred_action=current_action,
            )
            improved[state] = improved_action
            stable = stable and improved_action == current_action
        policy = improved
        if stable:
            return PlanningResult(values=values, policy=policy, iterations=iteration)
    raise RuntimeError("Policy iteration did not converge within the iteration limit.")


def values_as_array(
    environment: StochasticGridworld, values: Mapping[State, float]
) -> np.ndarray:
    return np.array(
        [values[state] for state in environment.states], dtype=float
    ).reshape(environment.rows, environment.columns)


def policy_as_array(
    environment: StochasticGridworld, policy: Mapping[State, Action | None]
) -> np.ndarray:
    return np.array(
        [
            policy[state] if policy[state] is not None else "G"
            for state in environment.states
        ],
        dtype=object,
    ).reshape(environment.rows, environment.columns)
