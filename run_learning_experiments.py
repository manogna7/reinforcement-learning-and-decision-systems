from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from decision_making import (
    temporal_difference_gridworld,
    train_actor_critic,
    train_q_learning,
    train_sarsa,
    train_sarsa_lambda,
)

Trainer = Callable[..., Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare online control methods in a stochastic gridworld."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("artifacts/temporal-difference-learning-curves.png"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def _run_trials(
    trainer: Trainer,
    *,
    trials: int,
    episodes: int,
    seed: int,
    max_episode_steps: int,
    parameters: dict[str, float],
) -> tuple[np.ndarray, int]:
    environment = temporal_difference_gridworld()
    reward_matrix = np.zeros((trials, episodes), dtype=float)
    truncated_episodes = 0
    for trial in range(trials):
        result = trainer(
            environment,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            seed=seed + trial,
            **parameters,
        )
        reward_matrix[trial] = result.episode_rewards
        truncated_episodes += result.truncated_episodes
    return reward_matrix, truncated_episodes


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _save_plot(results: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((path.parent / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    for name, reward_matrix in results.items():
        mean = reward_matrix.mean(axis=0)
        deviation = reward_matrix.std(axis=0)
        smoothed_mean = _moving_average(mean)
        smoothed_deviation = _moving_average(deviation)
        episodes = np.arange(len(smoothed_mean)) + (len(mean) - len(smoothed_mean)) + 1
        axis.plot(episodes, smoothed_mean, label=name)
        axis.fill_between(
            episodes,
            smoothed_mean - smoothed_deviation,
            smoothed_mean + smoothed_deviation,
            alpha=0.14,
        )
    axis.set_title("Online reinforcement-learning comparison")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Total reward")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episodes <= 0 or args.trials <= 0:
        raise SystemExit("Episodes and trials must be positive.")

    configurations: dict[str, tuple[Trainer, dict[str, float]]] = {
        "SARSA": (
            train_sarsa,
            {"learning_rate": 0.1, "epsilon": 0.05},
        ),
        "Q-learning": (
            train_q_learning,
            {"learning_rate": 0.05, "epsilon": 0.01},
        ),
        "SARSA(lambda)": (
            train_sarsa_lambda,
            {
                "learning_rate": 0.05,
                "epsilon": 0.05,
                "trace_decay": 0.7,
            },
        ),
        "Actor-critic": (
            train_actor_critic,
            {
                "actor_learning_rate": 0.1,
                "critic_learning_rate": 0.001,
                "epsilon": 0.01,
            },
        ),
    }
    results: dict[str, np.ndarray] = {}
    for offset, (name, (trainer, parameters)) in enumerate(configurations.items()):
        rewards, truncated = _run_trials(
            trainer,
            trials=args.trials,
            episodes=args.episodes,
            seed=args.seed + offset * 10_000,
            max_episode_steps=args.max_episode_steps,
            parameters=parameters,
        )
        results[name] = rewards
        final_rewards = rewards[:, -1]
        print(
            f"{name}: final reward={final_rewards.mean():.2f} "
            f"+/- {final_rewards.std():.2f}; "
            f"truncated episodes={truncated}"
        )

    if not args.no_plot:
        _save_plot(results, args.plot)
        print(f"Learning curves: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
