import numpy as np
import random
import functools


class Game:
    def __init__(self, N=4):
        self.N = N
        self.board = np.zeros((N, N), dtype=int)
        self._pop_up_number()
        self._pop_up_number()
        self.score = 0
        self.is_game_over = False

    def move(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            # before the func call
            print(func.__name__)

            # func call
            ret = func(self, *args, **kwargs)

            # after the func call
            self.calculate_score()
            print(self.board)
            self.game_over_check()
            if self.is_game_over:
                print('Game over! Score:', self.score)
            return ret

        return _decorator

    @move
    def move_up(self):
        new_board = np.copy(self.board)
        new_board = np.apply_along_axis(self._move_array_up, 0, new_board)
        # merge same neighbors
        # iterate through rows
        for i in range(self.N - 1):
            merge_indices = new_board[i, :] == new_board[i + 1, :]
            new_board[i, merge_indices] += new_board[i + 1, merge_indices]
            new_board[i + 1, merge_indices] = 0
        new_board = np.apply_along_axis(self._move_array_up, 0, new_board)
        if np.all(new_board == self.board):
            print('Move did not change the board, no new number popped up.')
        else:
            self.board = new_board
            self._pop_up_number()

    @move
    def move_down(self):
        new_board = np.copy(self.board)
        new_board = np.apply_along_axis(self._move_array_down, 0, new_board)
        # merge same neighbors
        # iterate through rows
        for i in reversed(range(1, self.N)):
            merge_indices = new_board[i, :] == new_board[i - 1, :]
            new_board[i, merge_indices] += new_board[i - 1, merge_indices]
            new_board[i - 1, merge_indices] = 0
        new_board = np.apply_along_axis(self._move_array_down, 0, new_board)
        if np.all(new_board == self.board):
            print('Move did not change the board, no new number popped up.')
        else:
            self.board = new_board
            self._pop_up_number()

    @move
    def move_left(self):
        new_board = np.copy(self.board)
        new_board = np.apply_along_axis(self._move_array_left, 1, new_board)
        # merge same neighbors
        # iterate through columns
        for i in reversed(range(1, self.N)):
            merge_indices = new_board[:, i] == new_board[:, i - 1]
            new_board[merge_indices, i] += new_board[merge_indices, i - 1]
            new_board[merge_indices, i - 1] = 0
        new_board = np.apply_along_axis(self._move_array_left, 1, new_board)
        if np.all(new_board == self.board):
            print('Move did not change the board, no new number popped up.')
        else:
            self.board = new_board
            self._pop_up_number()

    @move
    def move_right(self):
        new_board = np.copy(self.board)
        new_board = np.apply_along_axis(self._move_array_right, 1, new_board)
        # merge same neighbors
        # iterate through columns
        for i in range(self.N - 1):
            merge_indices = new_board[:, i] == new_board[:, i + 1]
            new_board[merge_indices, i] += new_board[merge_indices, i + 1]
            new_board[merge_indices, i + 1] = 0
        new_board = np.apply_along_axis(self._move_array_right, 1, new_board)
        if np.all(new_board == self.board):
            print('Move did not change the board, no new number popped up.')
        else:
            self.board = new_board
            self._pop_up_number()

    @staticmethod
    def _move_array_down(a):
        return np.concatenate([a[a == 0], a[a != 0]])

    @staticmethod
    def _move_array_up(a):
        return np.concatenate([a[a != 0], a[a == 0]])

    @staticmethod
    def _move_array_left(a):
        return np.concatenate([a[a != 0], a[a == 0]])

    @staticmethod
    def _move_array_right(a):
        return np.concatenate([a[a == 0], a[a != 0]])

    @staticmethod
    def _add_right(a):
        diffs = np.ediff1d(a)
        return np.concatenate([a[a == 0], a[a != 0]])

    def calculate_score(self):
        self.score = self.board.sum()  # TODO: change to actual way to score

    def _pop_up_number(self):
        print('pop up number')
        pop_up_number = 2  # TODO: replace this by random choice between 2 and 4
        # choose random index where board is zero
        free_indices = np.argwhere(self.board == 0)
        pop_up_index = free_indices[np.random.choice(free_indices.shape[0])]
        self.board[pop_up_index[0], pop_up_index[1]] = pop_up_number

    def _random_moves(self, n_moves):
        moves = [self.move_left, self.move_right, self.move_up, self.move_down]
        for _ in range(n_moves):
            random.choice(moves)()
            if self.is_game_over:
                break

    def game_over_check(self):
        if np.any(self.board == 0):
            self.is_game_over = False
        elif np.any(np.diff(self.board, n=1, axis=0) == 0) or np.any(np.diff(self.board, n=1, axis=1) == 0):
            self.is_game_over = False
        else:
            self.is_game_over = True
        return self.is_game_over


class GameSimulator:
    def __init__(self, n_runs=1):
        self.n_runs = n_runs
        self.scores = np.zeros(n_runs)

    def simulate_strategy(self, strategy):
        if strategy == 'swirl':
            for i in range(self.n_runs):
                self.scores[i] = self._swirl_strategy()

    def summary(self):
        print('avg: {}'.format(self.scores.mean()))
        print('std: {}'.format(self.scores.std()))
        print('min: {}'.format(self.scores.min()))
        print('max: {}'.format(self.scores.max()))

    def _swirl_strategy(self):
        g = Game()
        i = 0
        move_list = [g.move_down, g.move_left, g.move_up, g.move_right]
        while not g.is_game_over:
            move = move_list[i % 4]
            move()
            i += 1
        return g.score
