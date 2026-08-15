from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

State: TypeAlias = tuple[int, int]
Action: TypeAlias = str
ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")
ACTION_EFFECTS: dict[Action, State] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_ACTIONS: dict[Action, tuple[Action, Action]] = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}


@dataclass(frozen=True)
class Transition:
    probability: float
    next_state: State
    reward: float
    terminated: bool


@dataclass(frozen=True)
class StochasticGridworld:
    rows: int = 4
    columns: int = 4
    start_state: State = (0, 0)
    goal_state: State = (3, 3)
    water_cells: frozenset[State] = field(
        default_factory=lambda: frozenset({(1, 1), (2, 2)})
    )
    wildfire_cells: frozenset[State] = field(
        default_factory=lambda: frozenset({(0, 3), (3, 0)})
    )
    reward_goal: float = 100.0
    reward_water: float = -5.0
    reward_wildfire: float = -10.0
    reward_default: float = -1.0
    success_probability: float = 0.8
    slide_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Grid dimensions must be positive.")
        if not self.in_bounds(self.start_state) or not self.in_bounds(self.goal_state):
            raise ValueError("Start and goal states must be inside the grid.")
        hazard_cells = self.water_cells | self.wildfire_cells
        if any(not self.in_bounds(state) for state in hazard_cells):
            raise ValueError("Every hazard cell must be inside the grid.")
        if self.water_cells & self.wildfire_cells:
            raise ValueError("Water and wildfire cells cannot overlap.")
        if self.start_state in hazard_cells or self.goal_state in hazard_cells:
            raise ValueError("Start and goal states cannot be hazards.")
        probability_mass = self.success_probability + 2 * self.slide_probability
        if not math.isclose(probability_mass, 1.0, abs_tol=1e-12):
            raise ValueError("Intended and slide probabilities must sum to one.")
        if self.success_probability < 0 or self.slide_probability < 0:
            raise ValueError("Transition probabilities cannot be negative.")

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(
            (row, column) for row in range(self.rows) for column in range(self.columns)
        )

    @property
    def actions(self) -> tuple[Action, ...]:
        return ACTIONS

    def in_bounds(self, state: State) -> bool:
        row, column = state
        return 0 <= row < self.rows and 0 <= column < self.columns

    def reward_for(self, state: State) -> float:
        if state == self.goal_state:
            return self.reward_goal
        if state in self.water_cells:
            return self.reward_water
        if state in self.wildfire_cells:
            return self.reward_wildfire
        return self.reward_default

    def move(self, state: State, action: Action) -> State:
        if action not in ACTION_EFFECTS:
            raise ValueError(f"Unknown action: {action}")
        row_delta, column_delta = ACTION_EFFECTS[action]
        candidate = (state[0] + row_delta, state[1] + column_delta)
        return candidate if self.in_bounds(candidate) else state

    def transitions(self, state: State, action: Action) -> tuple[Transition, ...]:
        if not self.in_bounds(state):
            raise ValueError(f"State is outside the grid: {state}")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        if state == self.goal_state:
            return (Transition(1.0, state, 0.0, True),)

        outcomes: defaultdict[State, float] = defaultdict(float)
        outcomes[self.move(state, action)] += self.success_probability
        for side_action in SIDE_ACTIONS[action]:
            outcomes[self.move(state, side_action)] += self.slide_probability

        transitions = tuple(
            Transition(
                probability=probability,
                next_state=next_state,
                reward=self.reward_for(next_state),
                terminated=next_state == self.goal_state,
            )
            for next_state, probability in sorted(outcomes.items())
        )
        if not math.isclose(
            sum(transition.probability for transition in transitions),
            1.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Transition construction lost probability mass.")
        return transitions

    def sample_transition(
        self, state: State, action: Action, rng: np.random.Generator
    ) -> Transition:
        transitions = self.transitions(state, action)
        probabilities = [transition.probability for transition in transitions]
        index = int(rng.choice(len(transitions), p=probabilities))
        return transitions[index]


def default_gridworld() -> StochasticGridworld:
    return StochasticGridworld()
