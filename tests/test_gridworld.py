import unittest

import numpy as np

from decision_making.gridworld import default_gridworld


class GridworldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = default_gridworld()

    def test_corner_transitions_consolidate_duplicate_outcomes(self) -> None:
        transitions = self.environment.transitions((0, 0), "up")
        probabilities = {
            transition.next_state: transition.probability for transition in transitions
        }
        self.assertEqual(probabilities, {(0, 0): 0.9, (0, 1): 0.1})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_interior_transition_rewards_follow_successor_state(self) -> None:
        transitions = self.environment.transitions((1, 2), "left")
        outcomes = {
            transition.next_state: (transition.probability, transition.reward)
            for transition in transitions
        }
        self.assertEqual(outcomes[(1, 1)], (0.8, -5.0))
        self.assertEqual(outcomes[(0, 2)], (0.1, -1.0))
        self.assertEqual(outcomes[(2, 2)], (0.1, -5.0))

    def test_goal_is_absorbing_without_repeating_goal_reward(self) -> None:
        transition = self.environment.transitions(self.environment.goal_state, "up")[0]
        self.assertEqual(transition.next_state, self.environment.goal_state)
        self.assertEqual(transition.probability, 1.0)
        self.assertEqual(transition.reward, 0.0)
        self.assertTrue(transition.terminated)

    def test_seeded_sampling_is_reproducible(self) -> None:
        first_rng = np.random.default_rng(7)
        second_rng = np.random.default_rng(7)
        first = [
            self.environment.sample_transition((0, 0), "right", first_rng).next_state
            for _ in range(20)
        ]
        second = [
            self.environment.sample_transition((0, 0), "right", second_rng).next_state
            for _ in range(20)
        ]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
