import unittest

import numpy as np

from decision_making.gridworld import default_gridworld
from decision_making.planning import (
    policy_evaluation_exact,
    policy_evaluation_iterative,
    policy_iteration,
    value_iteration,
    values_as_array,
)


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = default_gridworld()

    def test_value_iteration_reproduces_the_reference_value_function(self) -> None:
        result = value_iteration(self.environment, discount=0.95)
        expected = np.array(
            [
                [64.10, 69.02, 74.58, 80.55],
                [69.02, 75.61, 81.93, 89.10],
                [74.58, 81.93, 89.86, 97.17],
                [80.55, 89.10, 97.17, 0.00],
            ]
        )
        np.testing.assert_allclose(
            values_as_array(self.environment, result.values), expected, atol=0.01
        )

    def test_low_discount_reference_value_function(self) -> None:
        result = value_iteration(self.environment, discount=0.3)
        expected = np.array(
            [
                [-1.42, -1.41, -0.92, 2.84],
                [-1.41, 0.01, 3.90, 19.51],
                [-0.92, 3.90, 21.40, 82.52],
                [2.84, 19.51, 82.52, 0.00],
            ]
        )
        np.testing.assert_allclose(
            values_as_array(self.environment, result.values), expected, atol=0.01
        )

    def test_policy_iteration_and_value_iteration_agree(self) -> None:
        value_result = value_iteration(self.environment, discount=0.95)
        policy_result = policy_iteration(self.environment, discount=0.95)
        np.testing.assert_allclose(
            values_as_array(self.environment, policy_result.values),
            values_as_array(self.environment, value_result.values),
            atol=1e-7,
        )
        self.assertEqual(policy_result.policy, value_result.policy)

    def test_exact_and_iterative_policy_evaluation_agree(self) -> None:
        policy = value_iteration(self.environment, discount=0.95).policy
        exact = policy_evaluation_exact(self.environment, policy, discount=0.95)
        iterative = policy_evaluation_iterative(
            self.environment, policy, discount=0.95
        ).values
        np.testing.assert_allclose(
            [exact[state] for state in self.environment.states],
            [iterative[state] for state in self.environment.states],
            atol=1e-8,
        )

    def test_invalid_discount_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            value_iteration(self.environment, discount=1.0)


if __name__ == "__main__":
    unittest.main()
