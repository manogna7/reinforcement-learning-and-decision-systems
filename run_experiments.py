from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from decision_making import (
    default_gridworld,
    policy_iteration,
    run_dagger,
    simulate_episode,
    value_iteration,
)
from decision_making.planning import policy_as_array, values_as_array


def _parse_checkpoints(value: str) -> tuple[int, ...]:
    try:
        checkpoints = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Checkpoints must be comma-separated integers."
        ) from error
    if not checkpoints or any(checkpoint <= 0 for checkpoint in checkpoints):
        raise argparse.ArgumentTypeError("Checkpoints must be positive integers.")
    return checkpoints


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run planning and imitation-learning experiments on a stochastic grid."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dagger-checkpoints",
        type=_parse_checkpoints,
        default=(5, 10, 20, 30, 40, 50),
    )
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("artifacts/dagger-accuracy.png"),
        help="Output path for the DAgger learning curve.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def _save_plot(iterations: list[int], accuracies: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((path.parent / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(iterations, accuracies, marker="o")
    axis.set_title("DAgger policy agreement")
    axis.set_xlabel("Aggregation iterations")
    axis.set_ylabel("Non-terminal state accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    environment = default_gridworld()

    for discount in (0.3, 0.95):
        result = value_iteration(environment, discount=discount)
        print(
            f"\nValue iteration (discount={discount}, iterations={result.iterations})"
        )
        print(policy_as_array(environment, result.policy))
        print(values_as_array(environment, result.values).round(2))

    expert = value_iteration(environment, discount=0.95)
    policy_result = policy_iteration(environment, discount=0.95)
    print(f"\nPolicy iteration (iterations={policy_result.iterations})")
    print(policy_as_array(environment, policy_result.policy))
    print(values_as_array(environment, policy_result.values).round(2))

    import numpy as np

    episode = simulate_episode(
        environment,
        expert.policy,
        np.random.default_rng(args.seed),
        max_steps=args.max_episode_steps,
    )
    print(
        f"\nExpert rollout: steps={len(episode.steps)}, "
        f"reward={episode.total_reward:.1f}, reached_goal={episode.reached_goal}"
    )

    dagger = run_dagger(
        environment,
        expert.policy,
        checkpoints=args.dagger_checkpoints,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    print(f"Aggregated examples: {dagger.aggregated_examples}")
    for metric in dagger.metrics:
        print(
            f"DAgger iteration={metric.iteration:>2} "
            f"accuracy={metric.accuracy:.3f} "
            f"rollout_reward={metric.rollout_reward:.1f} "
            f"reached_goal={metric.reached_goal}"
        )

    if not args.no_plot:
        _save_plot(
            [metric.iteration for metric in dagger.metrics],
            [metric.accuracy for metric in dagger.metrics],
            args.plot,
        )
        print(f"Learning curve: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
