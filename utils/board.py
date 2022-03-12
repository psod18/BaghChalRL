from typing import Type

import numpy as np

from agents import (
    Sheep,
    Wolves,
)


class BaghChal:
    def __init__(self, sheep_agent_cls: Type[Sheep], wolves_agent_cls: Type[Wolves]):
        self.board = self._create_board()

        # TODO: warning if agents are not proper subclasses
        self.sheep = sheep_agent_cls(self.board)
        self.wolves = wolves_agent_cls(self.board)


    @staticmethod
    def _create_board():
        """Initiate game board and place wolves pieces to each corner"""
        board = np.zeros((5, 5), dtype=np.int32)
        return board

    def play_match(self):
        s_count = w_count = 0
        while True:
            self.sheep.set_state(
                self.sheep.pick_state(
                    self.sheep.get_states()
                )
            )
            s_count += 1
            w_states = self.wolves.get_states()
            if not w_states:
                print("Wolves don't have options to make they turn. Sheep wins!")
                break
            self.wolves.set_state(
                self.wolves.pick_state(w_states)
            )
            if self.wolves.captured_sheep == 5:
                print(f"Wolves ate {self.wolves.captured_sheep} sheep already. Wolves wins!")
                break
            w_count += 1

        print(f"Sheep turns: {s_count}, Wolves turns: {w_count}")
        print(f"Captured sheep: {self.wolves.captured_sheep}")
        print('----- final board state -----')
        print(self.board)


if __name__ == "__main__":
    t = BaghChal(Sheep, Wolves)
    t.play_match()

