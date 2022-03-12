from abc import abstractmethod
from typing import List

import numpy as np

"""
Field is 5x5 nodes connected to each neighbors:
 - one-to-six if i,j are odd row (except edge node);
 - one-to-fore if i,j are even (except edge node).
 O -- O -- O -- O -- O
 |  \ |  / |  \ |  / |
 O -- O -- O -- O -- O
 |  / |  \ |  / |  \ |
 O -- O -- O -- O -- O
 |  \ |  / |  \ |  / |
 O -- O -- O -- O -- O
 |  / |  \ |  / |  \ |
 O -- O -- O -- O -- O
"""


class BaghChal:
    def __init__(self):
        self.board = self._create_board()
        self.sheep_agent = Sheep(self.board)
        self.wolves_agent = Wolves(self.board)

        self.game_is_over = False

    @staticmethod
    def _create_board():
        """Initiate game board and place wolves pieces to each corner"""
        board = np.zeros((5, 5), dtype=np.int32)
        return board


class Player:

    def __init__(self, board: np.array):
        self.board = board

    @abstractmethod
    def get_states(self) -> List[np.array]:
        """
        Return tuple of available moves.
        0 - empty field
        1 - field occupied with sheep
       -1 - field occupied with wolves
        :return: list of next possible states (state0, state1, ...)
        """

    @abstractmethod
    def set_state(self, state: np.array) -> None:
        """

        :param state:
        """

    @abstractmethod
    def _extract_pieces_position(self) -> np.array:
        """get array of x,y coordinates for all player figures"""


class Sheep(Player):
    """figures are marked with 1 in board array"""

    def __init__(self, board):
        super(Sheep, self).__init__(board)
        self.in_reserve = 20
        self.color = "white"

    def get_states(self):
        """
        Generate all possible next board states
        :return: list of states -  List[array(5x5)]
        """
        if self.in_reserve > 0:
            _positions = self._get_empty_fields()
            new_states = self._generate_states(_positions, add_new=True)
        else:
            _positions = self._extract_pieces_position()
            new_states = self._generate_states(_positions)
        return new_states

    def set_state(self, new_state: np.array) -> None:
        if (self.board-new_state).sum() < 0:
            # new sheep was add to the board
            self.in_reserve -= 1

        self.board[:] = new_state[:]

    def _extract_pieces_position(self) -> np.array:
        return np.array(np.where(self.board > 0)).T

    def _generate_states(self, positions: np.array, add_new: bool = False) -> np.array:
        _states = []
        if add_new:
            for pos in positions:
                _states.append(self._create_state(None, pos))
        else:
            for pos in positions:
                available_tails = self._get_available_moves(*pos)
                for target in available_tails:
                    _states.append(self._create_state(pos, target))
        return _states

    def _create_state(self, start_tail, end_tail):
        board = np.copy(self.board)
        if start_tail:
            x0, y0 = start_tail
            board[x0][y0] = 0
        x, y = end_tail
        board[x][y] = 1
        return board

    def _get_empty_fields(self) -> np.array:
        return np.array(np.where(self.board == 0)).T

    def _get_available_moves(self, x, y):
        tails = []

        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if (x + y) % 2 == 0:
                    if all([4 >= l >= 0 for l in (i, j)]) and (x, y) != (i, j):
                        if self.board[i][j] == 0:
                            tails.append([i, j])
                else:
                    if all([4 >= l >= 0 for l in (i, j)]) and (x, y) != (i, j) and (i == x or j == y):
                        if self.board[i][j] == 0:
                            tails.append([i, j])
        return tails


class Wolves(Player):
    """figures are marked with -1 in board array"""

    def __init__(self, board):
        super().__init__(board)
        self.captured_sheep = 0
        self.color = "black"

        # Place wolves in the corner of the game board:
        self.board[[0, 0, 4, 4], [0, 4, 0, 4]] = -1

    def get_states(self):
        _positions = self._extract_pieces_position()
        _states = self._generate_states(_positions)
        return _states

    def set_state(self, new_state: np.array) -> None:
        if (self.board - new_state).sum() > 0:
            # one sheep was captured at this turn
            self.captured_sheep += 1

        self.board[:] = new_state[:]

    def _extract_pieces_position(self) -> np.array:
        return np.array(np.where(self.board < 0)).T

    def _generate_states(self, positions: np.array):
        _states = []
        for pos in positions:
            available_tails = self._get_available_moves(*pos)
            for target, prey in available_tails:
                _states.append(self._create_state(pos, target, prey))
        return _states

    def _get_available_moves(self, x, y):
        tails = []

        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if (x + y) % 2 == 0:
                    if all([4 >= k >= 0 for k in (i, j)]) and (x, y) != (i, j):
                        if self.board[i][j] == 0:
                            tails.append(([i, j], None))
                        elif self.board[i][j] == 1:  # with sheep figure
                            l = i + (i - x)
                            m = j + (j - y)
                            if all([4 >= z >= 0 for z in (l, m)]) and self.board[i][j] == 0:
                                tails.append(([l, m], [i, j]))
                else:
                    if all([4 >= l >= 0 for l in (i, j)]) and (x, y) != (i, j) and (i == x or j == y):
                        if self.board[i][j] == 0:
                            tails.append([i, j])
                        elif self.board[i][j] == 1:  # with sheep figure
                            l = i + (i - x)
                            m = j + (j - y)
                            if all([4 >= z >= 0 for z in (l, m)]) and self.board[i][j] == 0:
                                tails.append(([l, m], [i, j]))
        return tails

    def _create_state(self, start_tail, end_tail, prey_tail):
        board = np.copy(self.board)
        if prey_tail:
            xp, yp = prey_tail
            board[xp][yp] = 0
        x0, y0 = start_tail
        board[x0][y0] = 0
        x, y = end_tail
        board[x][y] = -1
        return board


if __name__ == "__main__":
    t = BaghChal()
    print(t.board)
    print("Sheep states:")
    # for s in t.sheep_agent.get_states(): print(s)
    print(t.sheep_agent.get_states()[-1])
    print("Wolves states:")
    # for s in t.wolves_agent.get_states(): print(s)
    print(t.wolves_agent.get_states()[-1])
