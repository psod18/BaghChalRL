from typing import List

import numpy as np

from utils.agents.base import Sheep, Wolves


class GreedySheep(Sheep):
    """Greedy approach"""

    def pick_state(self, states: List[np.array]) -> np.array:
        return states[np.random.choice(np.arange(len(states)))]


class GreedyWolves(Wolves):
    """Greedy approach"""

    def pick_state(self, states: List[np.array]) -> np.array:
        return states[np.random.choice(np.arange(len(states)))]
