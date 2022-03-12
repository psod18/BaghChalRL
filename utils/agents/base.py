from abc import abstractmethod
from typing import List

import numpy as np


class Player:
    # self.__bases__[0].__name__ --> 'Player'

    def __init__(self, board: np.array):
        self.board = board

    @abstractmethod
    def get_states(self) -> List[np.array]:
        """
        Return tuple of available moves.
        0 - empty field
        1 - field occupied with sheep
       -1 - field occupied with wolf
        :return: list of next possible states (state0, state1, ...) - states are numpy arrays (5x5)
        """

    @abstractmethod
    def pick_state(self, states: List[np.array]) -> np.array:
        """
        Pick one board state as next state (can be understood as player move)
        :param states: list of all available next states
        :return: next state chosen by agent
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

    def pick_state(self, states: List[np.array]) -> np.array:
        """picks randomly next state (turn)"""
        return states[np.random.choice(np.arange(len(states)))]

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
        if start_tail is not None:
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

    def pick_state(self, states: List[np.array]) -> np.array:
        """picks randomly next state (turn)"""
        return states[np.random.choice(np.arange(len(states)))]

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

    def _get_available_moves(self, x0, y0):
        tails = []

        for x in range(x0 - 1, x0 + 2):
            if not 4 >= x >= 0:
                # out of board size
                continue
            for y in range(y0 - 1, y0 + 2):
                if not 4 >= y >= 0:
                    # out of board size
                    continue
                if x0 == x and y0 == y:
                    # initial tail
                    continue
                if (x0 + y0) % 2 == 0:
                    if self.board[x][y] == 0:
                        tails.append(([x, y], None))
                    elif self.board[x][y] == 1:  # with sheep figure
                        # check tail behind sheep
                        xp = x + (x - x0)
                        yp = y + (y - y0)
                        if all([4 >= z >= 0 for z in (xp, yp)]) and self.board[xp][yp] == 0:
                            tails.append(([xp, yp], [x, y]))
                else:
                    #
                    if x != x0 and y != y0:
                        # exclude diagonal direction
                        continue
                    if self.board[x][y] == 0:
                        tails.append(([x, y], None))
                    elif self.board[x][y] == 1:  # with sheep figure
                        # check tail behind sheep
                        xp = x + (x - x0)
                        yp = y + (y - y0)
                        if all([4 >= z >= 0 for z in (xp, yp)]) and self.board[xp][yp] == 0:
                            tails.append(([xp, yp], [x, y]))
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
