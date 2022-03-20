from typing import Type

import numpy as np

from agents import (
    Sheep,
    Wolves,
)


class BaghChal:
    def __init__(self, sheep_agent_cls: Type[Sheep], wolves_agent_cls: Type[Wolves]):
        self.board = self._create_board()

        self.sheep = sheep_agent_cls()
        self.wolves = wolves_agent_cls()

        self.done = False

    @staticmethod
    def _create_board():
        """Initiate game board and place wolves pieces to each corner"""
        board = np.zeros((5, 5), dtype=np.int32)
        return board

    def play_match(self):
        _round = 1
        while not self.done:

            for agent in (self.wolves, self.sheep):
                new_state, self.done, reward = agent.make_turn(self.board)
                if self.done:
                    print(f"{agent.__class__.__name__} lose!")
                    break

                self.board = new_state
            _round += 1

        print(f"Num of turns: {_round}")
        print(f"Captured sheep: {self.wolves.captured_sheep}")
        print('----- final board state -----')
        print(self.board)
        print((self.board>0).sum())


if __name__ == "__main__":
    t = BaghChal(Sheep, Wolves)
    t.play_match()

