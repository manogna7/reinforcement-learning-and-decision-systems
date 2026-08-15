import unittest

import numpy as np

from decision_making.gridworld import (
    StochasticGridworld,
    temporal_difference_gridworld,
)
from decision_making.temporal_difference import (
    actor_action_probabilities,
    state_features,
    train_actor_critic,
    train_q_learning,
    train_sarsa,
    train_sarsa_lambda,
)


class TemporalDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = temporal_difference_gridworld()

    def test_environment_matches_the_online_learning_experiment(self) -> None:
        self.assertEqual(
            self.environment.water_cells,
            frozenset({(1, 1), (1, 2)}),
        )
        self.assertEqual(
            self.environment.wildfire_cells,
            frozenset({(0, 2), (0, 3)}),
        )
        transitions = self.environment.transitions((0, 0), "up")
        self.assertEqual(
            {item.next_state: item.probability for item in transitions},
            {(0, 0): 0.9, (0, 1): 0.1},
        )

    def test_tabular_control_methods_are_seeded_and_bounded(self) -> None:
        trainers = (train_sarsa, train_q_learning, train_sarsa_lambda)
        for trainer in trainers:
            with self.subTest(trainer=trainer.__name__):
                first = trainer(
                    self.environment,
                    episodes=12,
                    max_episode_steps=80,
                    seed=23,
                )
                second = trainer(
                    self.environment,
                    episodes=12,
                    max_episode_steps=80,
                    seed=23,
                )
                np.testing.assert_array_equal(first.q_values, second.q_values)
                self.assertEqual(first.episode_rewards, second.episode_rewards)
                self.assertEqual(first.episode_lengths, second.episode_lengths)
                self.assertTrue(np.isfinite(first.q_values).all())
                self.assertTrue(all(length <= 80 for length in first.episode_lengths))
                self.assertIsNone(first.policy[self.environment.goal_state])

    def test_unreachable_goal_is_reported_as_truncated(self) -> None:
        result = train_q_learning(
            self.environment,
            episodes=3,
            max_episode_steps=1,
            seed=5,
        )
        self.assertEqual(result.episode_lengths, (1, 1, 1))
        self.assertEqual(result.truncated_episodes, 3)

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        environment = StochasticGridworld(
            rows=1,
            columns=2,
            start_state=(0, 0),
            goal_state=(0, 1),
            water_cells=frozenset(),
            wildfire_cells=frozenset(),
            success_probability=1.0,
            slide_probability=0.0,
        )
        right = environment.actions.index("right")
        for trainer in (train_sarsa, train_q_learning, train_sarsa_lambda):
            with self.subTest(trainer=trainer.__name__):
                result = trainer(
                    environment,
                    episodes=1,
                    learning_rate=1.0,
                    epsilon=1.0,
                    epsilon_decay=1.0,
                    minimum_epsilon=1.0,
                    max_episode_steps=1,
                    seed=1,
                )
                self.assertEqual(result.q_values[0, 0, right], 100.0)
                self.assertEqual(result.truncated_episodes, 0)

    def test_actor_critic_is_reproducible_and_numerically_stable(self) -> None:
        first = train_actor_critic(
            self.environment,
            episodes=15,
            max_episode_steps=100,
            seed=31,
        )
        second = train_actor_critic(
            self.environment,
            episodes=15,
            max_episode_steps=100,
            seed=31,
        )
        np.testing.assert_array_equal(first.critic_weights, second.critic_weights)
        np.testing.assert_array_equal(
            first.actor_preferences,
            second.actor_preferences,
        )
        self.assertEqual(first.episode_rewards, second.episode_rewards)
        probabilities = actor_action_probabilities(
            first.actor_preferences[self.environment.start_state]
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue((probabilities > 0.0).all())
        self.assertTrue(np.isfinite(first.actor_preferences).all())

    def test_state_features_preserve_the_original_semantics(self) -> None:
        np.testing.assert_array_equal(
            state_features(self.environment, (1, 1)),
            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
