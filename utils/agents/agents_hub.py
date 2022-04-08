import os
import random
from typing import List, Union

import numpy as np

from utils.agents.base import Sheep, Wolves


class QSheep(Sheep):
    """Q-learning with eps-Greedy approach"""

    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, eps: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q_table = {}

        self.trajectory = []

    def pick_state(self, q_states: List[np.array]) -> int:
        """picks action"""
        if random.uniform(0, 1) < self.eps:
            action_idx = np.random.choice(np.arange(len(q_states)))
        else:
            action_idx = np.argmax(q_states)
        return action_idx

    def make_turn(self, current_state: np.array) -> Union[np.array, None]:

        if (current_state > 0).sum() + self.in_reserve < 16:
            # Sheep agent lose
            return None

        state_hash = hash(current_state.tobytes()) + self.in_reserve

        _states = self.get_states(current_state)

        if len(_states) == 0:
            # workaround for case when all sheep are blocked by wolves
            return []

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(len(_states), dtype=np.float64)
        q_values = self.q_table[state_hash]

        next_state_idx = self.pick_state(q_values)
        try:
            new_state = _states[next_state_idx]
        except IndexError as e:
            print(current_state)
            print(self.in_reserve)
            for s in _states: print(s)
            print(len(_states))
            print(q_values, len(q_values))
            raise e
        self.trajectory.append((state_hash, next_state_idx))

        return new_state

    def update_q_from_trajectory(self, reward):

        # TODO: revert for speed up the training (and the same for sheep agent!)
        for j in range(len(self.trajectory)):
            q_s, idx = self.trajectory[j]
            try:
                q_sp, _ = self.trajectory[j+1]
                self.q_table[q_s][idx] += self.alpha*(self.gamma * np.max(self.q_table[q_sp]) - self.q_table[q_s][idx])
            except IndexError:
                self.q_table[q_s][idx] += self.alpha*(self.gamma * reward - self.q_table[q_s][idx])

        self.trajectory = []

    def save(self, path, filename="sheep"):
        np.save(os.path.join(path, f"{filename}.npy"), self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)


class QWolves(Wolves):
    """Q-learning with eps-Greedy approach"""

    def __init__(self, alpha: float = 0.5, gamma: float = 1.0, eps: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q_table = {}

        self.trajectory = []

    def pick_state(self, q_states: List[np.array]) -> int:
        """picks action"""
        if random.uniform(0, 1) < self.eps:
            action_idx = np.random.choice(np.arange(len(q_states)))
        else:
            action_idx = np.argmax(q_states)
        return action_idx

    def make_turn(self, current_state: np.array) -> Union[np.array, None]:
        state_hash = hash(current_state.tobytes())
        _states = self.get_states(current_state)
        if not _states:
            return None

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(len(_states), dtype=np.float64)
        q_values = self.q_table[state_hash]

        next_state_idx = self.pick_state(q_values)
        new_state = _states[next_state_idx]
        if (current_state - new_state).sum() > 0:
            self.captured_sheep += 1

        self.trajectory.append((state_hash, next_state_idx))
        return new_state

    def update_q_from_trajectory(self, reward):

        for j in range(len(self.trajectory)):
            q_s, idx = self.trajectory[j]
            try:
                q_sp, _ = self.trajectory[j+1]
                self.q_table[q_s][idx] += self.alpha*(self.gamma * np.max(self.q_table[q_sp]) - self.q_table[q_s][idx])
            except IndexError:
                self.q_table[q_s][idx] += self.alpha*(self.gamma * reward - self.q_table[q_s][idx])

        self.trajectory = []

    def save(self, path, filename="wolves"):
        np.save(os.path.join(path, f"{filename}.npy"), self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)
