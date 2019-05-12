import numpy as np

from gym import Env

from gym_2048.envs.variable_names import UP, RIGHT, DOWN, LEFT


class Gym2048Env(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, N=4, verbose=False):
        self.N = N
        self.verbose = verbose
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.verbose:
            print(f"Move {action}")
        new_board = np.copy(self.board)
        if action == UP:
            new_board = np.apply_along_axis(self._move_array_up, 0, new_board)
            # merge same neighbors
            # iterate through rows
            for i in range(self.N - 1):
                merge_indices = new_board[i, :] == new_board[i + 1, :]
                new_board[i, merge_indices] += new_board[i + 1, merge_indices]
                new_board[i + 1, merge_indices] = 0
            new_board = np.apply_along_axis(self._move_array_up, 0, new_board)
        elif action == RIGHT:
            new_board = np.apply_along_axis(self._move_array_right, 1, new_board)
            # merge same neighbors
            # iterate through columns
            for i in range(self.N - 1):
                merge_indices = new_board[:, i] == new_board[:, i + 1]
                new_board[merge_indices, i] += new_board[merge_indices, i + 1]
                new_board[merge_indices, i + 1] = 0
            new_board = np.apply_along_axis(self._move_array_right, 1, new_board)
        elif action == DOWN:
            new_board = np.apply_along_axis(self._move_array_down, 0, new_board)
            # merge same neighbors
            # iterate through rows
            for i in reversed(range(1, self.N)):
                merge_indices = new_board[i, :] == new_board[i - 1, :]
                new_board[i, merge_indices] += new_board[i - 1, merge_indices]
                new_board[i - 1, merge_indices] = 0
            new_board = np.apply_along_axis(self._move_array_down, 0, new_board)
        elif action == LEFT:
            new_board = np.apply_along_axis(self._move_array_left, 1, new_board)
            # merge same neighbors
            # iterate through columns
            for i in reversed(range(1, self.N)):
                merge_indices = new_board[:, i] == new_board[:, i - 1]
                new_board[merge_indices, i] += new_board[merge_indices, i - 1]
                new_board[merge_indices, i - 1] = 0
            new_board = np.apply_along_axis(self._move_array_left, 1, new_board)
        else:
            raise ValueError(f"Illegal action {action} selected. Options: {UP}, {RIGHT}, {DOWN}, {LEFT}.")

        # pop up new number
        if np.all(new_board == self.board):
            print('Move did not change the board, no new number popped up.')
        else:
            self.board = new_board
            self._pop_up_number()

        # game over check
        done = self.game_over_check()
        observation = self.board
        reward = self.calculate_score()
        info = None

        # verbose output
        if self.verbose:
            print(self.board)
            if self.is_game_over:
                print(f"Game over! Score: {self.score}")

        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.board = np.zeros((self.N, self.N), dtype=int)
        self._pop_up_number()
        self._pop_up_number()
        self.score = 0
        self.is_game_over = False

        return self.board

    def _pop_up_number(self):
        if self.verbose:
            print('pop up number')
        pop_up_number = 2  # TODO: replace this by random choice between 2 and 4
        # choose random index where board is zero
        free_indices = np.argwhere(self.board == 0)
        pop_up_index = free_indices[np.random.choice(free_indices.shape[0])]
        self.board[pop_up_index[0], pop_up_index[1]] = pop_up_number

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

    def game_over_check(self):
        if np.any(self.board == 0):
            self.is_game_over = False
        elif np.any(np.diff(self.board, n=1, axis=0) == 0) or np.any(np.diff(self.board, n=1, axis=1) == 0):
            self.is_game_over = False
        else:
            self.is_game_over = True
        return self.is_game_over
