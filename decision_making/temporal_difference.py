from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_making.gridworld import State, StochasticGridworld
from decision_making.planning import Policy


@dataclass(frozen=True)
class ControlResult:
    q_values: np.ndarray
    policy: Policy
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    truncated_episodes: int


@dataclass(frozen=True)
class ActorCriticResult:
    critic_weights: np.ndarray
    actor_preferences: np.ndarray
    policy: Policy
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    truncated_episodes: int


def _validate_training_parameters(
    environment: StochasticGridworld,
    *,
    episodes: int,
    learning_rates: tuple[float, ...],
    discount: float,
    epsilon: float,
    epsilon_decay: float,
    minimum_epsilon: float,
    max_episode_steps: int,
) -> None:
    if environment.start_state == environment.goal_state:
        raise ValueError("Training requires different start and goal states.")
    if episodes <= 0:
        raise ValueError("The episode count must be positive.")
    if any(rate <= 0 for rate in learning_rates):
        raise ValueError("Learning rates must be positive.")
    if not 0.0 <= discount < 1.0:
        raise ValueError("The discount factor must satisfy 0 <= discount < 1.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between zero and one.")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("Epsilon decay must be in (0, 1].")
    if not 0.0 <= minimum_epsilon <= epsilon:
        raise ValueError("Minimum epsilon must be between zero and epsilon.")
    if max_episode_steps <= 0:
        raise ValueError("The rollout horizon must be positive.")


def _epsilon_for_episode(
    initial_epsilon: float,
    minimum_epsilon: float,
    epsilon_decay: float,
    episode: int,
) -> float:
    return max(minimum_epsilon, initial_epsilon * epsilon_decay**episode)


def _epsilon_greedy_action(
    action_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(action_values)))
    maximum = np.max(action_values)
    candidates = np.flatnonzero(np.isclose(action_values, maximum))
    return int(rng.choice(candidates))


def _new_q_values(environment: StochasticGridworld) -> np.ndarray:
    return np.zeros(
        (environment.rows, environment.columns, len(environment.actions)),
        dtype=float,
    )


def policy_from_q_values(
    environment: StochasticGridworld,
    q_values: np.ndarray,
) -> Policy:
    expected_shape = (
        environment.rows,
        environment.columns,
        len(environment.actions),
    )
    if q_values.shape != expected_shape:
        raise ValueError(f"Q-values must have shape {expected_shape}.")
    return {
        state: (
            None
            if state == environment.goal_state
            else environment.actions[int(np.argmax(q_values[state]))]
        )
        for state in environment.states
    }


def _control_result(
    environment: StochasticGridworld,
    q_values: np.ndarray,
    rewards: list[float],
    lengths: list[int],
    truncated_episodes: int,
) -> ControlResult:
    return ControlResult(
        q_values=q_values,
        policy=policy_from_q_values(environment, q_values),
        episode_rewards=tuple(rewards),
        episode_lengths=tuple(lengths),
        truncated_episodes=truncated_episodes,
    )


def train_sarsa(
    environment: StochasticGridworld,
    *,
    episodes: int = 100,
    learning_rate: float = 0.1,
    discount: float = 0.95,
    epsilon: float = 0.05,
    epsilon_decay: float = 0.99,
    minimum_epsilon: float = 0.01,
    max_episode_steps: int = 500,
    seed: int = 17,
) -> ControlResult:
    _validate_training_parameters(
        environment,
        episodes=episodes,
        learning_rates=(learning_rate,),
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        minimum_epsilon=minimum_epsilon,
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    q_values = _new_q_values(environment)
    rewards: list[float] = []
    lengths: list[int] = []
    truncated_episodes = 0

    for episode in range(episodes):
        exploration = _epsilon_for_episode(
            epsilon, minimum_epsilon, epsilon_decay, episode
        )
        state = environment.start_state
        action_index = _epsilon_greedy_action(q_values[state], exploration, rng)
        total_reward = 0.0
        terminated = False

        for step in range(1, max_episode_steps + 1):
            transition = environment.sample_transition(
                state, environment.actions[action_index], rng
            )
            total_reward += transition.reward
            if transition.terminated:
                target = transition.reward
                next_action_index = 0
            else:
                next_action_index = _epsilon_greedy_action(
                    q_values[transition.next_state], exploration, rng
                )
                target = (
                    transition.reward
                    + discount * q_values[transition.next_state][next_action_index]
                )
            q_values[state][action_index] += learning_rate * (
                target - q_values[state][action_index]
            )
            state = transition.next_state
            action_index = next_action_index
            if transition.terminated:
                terminated = True
                break

        rewards.append(total_reward)
        lengths.append(step)
        if not terminated:
            truncated_episodes += 1

    return _control_result(environment, q_values, rewards, lengths, truncated_episodes)


def train_q_learning(
    environment: StochasticGridworld,
    *,
    episodes: int = 100,
    learning_rate: float = 0.05,
    discount: float = 0.95,
    epsilon: float = 0.01,
    epsilon_decay: float = 0.99,
    minimum_epsilon: float = 0.01,
    max_episode_steps: int = 500,
    seed: int = 17,
) -> ControlResult:
    _validate_training_parameters(
        environment,
        episodes=episodes,
        learning_rates=(learning_rate,),
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        minimum_epsilon=minimum_epsilon,
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    q_values = _new_q_values(environment)
    rewards: list[float] = []
    lengths: list[int] = []
    truncated_episodes = 0

    for episode in range(episodes):
        exploration = _epsilon_for_episode(
            epsilon, minimum_epsilon, epsilon_decay, episode
        )
        state = environment.start_state
        total_reward = 0.0
        terminated = False

        for step in range(1, max_episode_steps + 1):
            action_index = _epsilon_greedy_action(q_values[state], exploration, rng)
            transition = environment.sample_transition(
                state, environment.actions[action_index], rng
            )
            total_reward += transition.reward
            continuation = (
                0.0
                if transition.terminated
                else discount * float(np.max(q_values[transition.next_state]))
            )
            target = transition.reward + continuation
            q_values[state][action_index] += learning_rate * (
                target - q_values[state][action_index]
            )
            state = transition.next_state
            if transition.terminated:
                terminated = True
                break

        rewards.append(total_reward)
        lengths.append(step)
        if not terminated:
            truncated_episodes += 1

    return _control_result(environment, q_values, rewards, lengths, truncated_episodes)


def train_sarsa_lambda(
    environment: StochasticGridworld,
    *,
    episodes: int = 100,
    learning_rate: float = 0.05,
    discount: float = 0.95,
    epsilon: float = 0.05,
    epsilon_decay: float = 0.99,
    minimum_epsilon: float = 0.01,
    trace_decay: float = 0.7,
    max_episode_steps: int = 500,
    seed: int = 17,
) -> ControlResult:
    if not 0.0 <= trace_decay <= 1.0:
        raise ValueError("Trace decay must be between zero and one.")
    _validate_training_parameters(
        environment,
        episodes=episodes,
        learning_rates=(learning_rate,),
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        minimum_epsilon=minimum_epsilon,
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    q_values = _new_q_values(environment)
    rewards: list[float] = []
    lengths: list[int] = []
    truncated_episodes = 0

    for episode in range(episodes):
        exploration = _epsilon_for_episode(
            epsilon, minimum_epsilon, epsilon_decay, episode
        )
        eligibility = np.zeros_like(q_values)
        state = environment.start_state
        action_index = _epsilon_greedy_action(q_values[state], exploration, rng)
        total_reward = 0.0
        terminated = False

        for step in range(1, max_episode_steps + 1):
            transition = environment.sample_transition(
                state, environment.actions[action_index], rng
            )
            total_reward += transition.reward
            if transition.terminated:
                target = transition.reward
                next_action_index = 0
            else:
                next_action_index = _epsilon_greedy_action(
                    q_values[transition.next_state], exploration, rng
                )
                target = (
                    transition.reward
                    + discount * q_values[transition.next_state][next_action_index]
                )
            temporal_difference_error = target - q_values[state][action_index]
            eligibility[state][action_index] += 1.0
            q_values += learning_rate * temporal_difference_error * eligibility
            eligibility *= discount * trace_decay
            state = transition.next_state
            action_index = next_action_index
            if transition.terminated:
                terminated = True
                break

        rewards.append(total_reward)
        lengths.append(step)
        if not terminated:
            truncated_episodes += 1

    return _control_result(environment, q_values, rewards, lengths, truncated_episodes)


def state_features(
    environment: StochasticGridworld,
    state: State,
) -> np.ndarray:
    if not environment.in_bounds(state):
        raise ValueError("Feature state must be inside the grid.")
    return np.array(
        [
            float(state[0]),
            float(state[1]),
            float(state in environment.water_cells),
            float(state in environment.wildfire_cells),
            float(state == environment.goal_state),
            1.0,
        ],
        dtype=float,
    )


def actor_action_probabilities(preferences: np.ndarray) -> np.ndarray:
    shifted = preferences - np.max(preferences)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def train_actor_critic(
    environment: StochasticGridworld,
    *,
    episodes: int = 100,
    actor_learning_rate: float = 0.1,
    critic_learning_rate: float = 0.001,
    discount: float = 0.95,
    epsilon: float = 0.01,
    epsilon_decay: float = 0.99,
    minimum_epsilon: float = 0.01,
    max_episode_steps: int = 500,
    seed: int = 17,
) -> ActorCriticResult:
    _validate_training_parameters(
        environment,
        episodes=episodes,
        learning_rates=(actor_learning_rate, critic_learning_rate),
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        minimum_epsilon=minimum_epsilon,
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    critic_weights = np.zeros(6, dtype=float)
    actor_preferences = np.zeros(
        (environment.rows, environment.columns, len(environment.actions)),
        dtype=float,
    )
    rewards: list[float] = []
    lengths: list[int] = []
    truncated_episodes = 0

    for episode in range(episodes):
        exploration = _epsilon_for_episode(
            epsilon, minimum_epsilon, epsilon_decay, episode
        )
        state = environment.start_state
        total_reward = 0.0
        terminated = False

        for step in range(1, max_episode_steps + 1):
            probabilities = actor_action_probabilities(actor_preferences[state])
            behavior_probabilities = (
                1.0 - exploration
            ) * probabilities + exploration / len(environment.actions)
            action_index = int(
                rng.choice(len(environment.actions), p=behavior_probabilities)
            )
            transition = environment.sample_transition(
                state, environment.actions[action_index], rng
            )
            total_reward += transition.reward

            features = state_features(environment, state)
            value = float(critic_weights @ features)
            next_value = (
                0.0
                if transition.terminated
                else float(
                    critic_weights @ state_features(environment, transition.next_state)
                )
            )
            temporal_difference_error = (
                transition.reward + discount * next_value - value
            )
            critic_weights += (
                critic_learning_rate * temporal_difference_error * features
            )

            policy_gradient = -probabilities
            policy_gradient[action_index] += 1.0
            actor_preferences[state] += (
                actor_learning_rate * temporal_difference_error * policy_gradient
            )

            state = transition.next_state
            if transition.terminated:
                terminated = True
                break

        rewards.append(total_reward)
        lengths.append(step)
        if not terminated:
            truncated_episodes += 1

    policy: Policy = {
        state: (
            None
            if state == environment.goal_state
            else environment.actions[int(np.argmax(actor_preferences[state]))]
        )
        for state in environment.states
    }
    return ActorCriticResult(
        critic_weights=critic_weights,
        actor_preferences=actor_preferences,
        policy=policy,
        episode_rewards=tuple(rewards),
        episode_lengths=tuple(lengths),
        truncated_episodes=truncated_episodes,
    )
