import unittest

import numpy as np

from run_learning_experiments import _summarize_final_window


class LearningExperimentTests(unittest.TestCase):
    def test_summary_aggregates_trailing_episodes_within_each_trial(self) -> None:
        rewards = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
            ]
        )
        mean, deviation, window = _summarize_final_window(rewards, 2)
        self.assertEqual(window, 2)
        self.assertEqual(mean, 5.25)
        self.assertEqual(deviation, 1.75)

    def test_summary_window_is_capped_by_available_episodes(self) -> None:
        rewards = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean, deviation, window = _summarize_final_window(rewards, 10)
        self.assertEqual(window, 2)
        self.assertEqual(mean, 2.5)
        self.assertEqual(deviation, 1.0)

    def test_summary_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            _summarize_final_window(np.array([]), 1)
        with self.assertRaises(ValueError):
            _summarize_final_window(np.ones((2, 2)), 0)


if __name__ == "__main__":
    unittest.main()
