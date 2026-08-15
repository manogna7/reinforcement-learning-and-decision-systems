import unittest

import numpy as np

from decision_making.gridworld import default_gridworld
from decision_making.imitation import policy_accuracy, run_dagger, simulate_episode
from decision_making.planning import value_iteration


class ImitationLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = default_gridworld()
        self.expert = value_iteration(self.environment, discount=0.95).policy

    def test_terminal_state_does_not_cap_policy_accuracy(self) -> None:
        candidate = self.expert.copy()
        candidate[self.environment.goal_state] = "up"
        self.assertEqual(policy_accuracy(self.environment, candidate, self.expert), 1.0)

    def test_rollout_horizon_prevents_nonterminating_policy(self) -> None:
        looping_policy = {
            state: None if state == self.environment.goal_state else "up"
            for state in self.environment.states
        }
        episode = simulate_episode(
            self.environment,
            looping_policy,
            np.random.default_rng(3),
            max_steps=12,
        )
        self.assertFalse(episode.reached_goal)
        self.assertEqual(len(episode.steps), 12)

    def test_dagger_is_reproducible_and_reports_requested_checkpoints(self) -> None:
        first = run_dagger(
            self.environment,
            self.expert,
            checkpoints=(2, 5, 10),
            seed=11,
            max_episode_steps=50,
        )
        second = run_dagger(
            self.environment,
            self.expert,
            checkpoints=(2, 5, 10),
            seed=11,
            max_episode_steps=50,
        )
        self.assertEqual(first, second)
        self.assertEqual([metric.iteration for metric in first.metrics], [2, 5, 10])
        self.assertTrue(all(0.0 <= metric.accuracy <= 1.0 for metric in first.metrics))
        self.assertGreater(first.aggregated_examples, 0)


if __name__ == "__main__":
    unittest.main()
