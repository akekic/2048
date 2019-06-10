import numpy as np

from gym import Env

from gym_2048.envs.variable_names import UP, RIGHT, DOWN, LEFT


class Gym2048Env(Env):
    metadata = {'render.modes': ['human']}
    action_space = [UP, RIGHT, DOWN, LEFT]

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
        new_board = self.slide_and_merge(new_board, action)

        # pop up new number
        if np.all(new_board == self.board):
            if self.verbose:
                print('Move did not change the board, no new number popped up.')
            reward = 0
        else:
            reward = self.calculate_reward(self.board, new_board)
            self.board = new_board
            self._pop_up_number()

        # game over check
        done = self.game_over_check()
        observation = self.board
        info = None

        # verbose output
        if self.verbose:
            print(self.board)
            if self.is_game_over:
                print(f"Game over! Score: {self.score}")

        return observation, reward, done, info

    def slide_and_merge(self, new_board, action):
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
        return new_board

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

    @staticmethod
    def calculate_reward(old_board, new_board):
        # count unique values in old and new board
        old_val, old_cts = np.unique(old_board, return_counts=True)
        old_value_counts = dict(zip(old_val, old_cts))

        new_val, new_cts = np.unique(new_board, return_counts=True)
        new_value_counts = dict(zip(new_val, new_cts))

        # extend value count dicts by values from other board
        for k_new in new_value_counts.keys():
            if k_new not in old_value_counts:
                old_value_counts[k_new] = 0

        for k_old in old_value_counts.keys():
            if k_old not in new_value_counts:
                new_value_counts[k_old] = 0

        reward = 0
        for val in reversed(np.sort(new_val)):
            if old_value_counts[val] < new_value_counts[val]:
                # count how often val is not yet accounted for
                n_diff = new_value_counts[val] - old_value_counts[val]
                reward += val * (n_diff)

                # account for fields that had to be merged
                new_value_counts[val / 2] += 2 * n_diff

        return reward

    def game_over_check(self):
        if np.any(self.board == 0):
            self.is_game_over = False
        elif np.any(np.diff(self.board, n=1, axis=0) == 0) or np.any(np.diff(self.board, n=1, axis=1) == 0):
            self.is_game_over = False
        else:
            self.is_game_over = True
        return self.is_game_over

    def next_states(self, action):
        """
        Returns possible next states if action was taken. All states have equal probability.
        """
        current_board = np.copy(self.board)
        board_after_merge = self.slide_and_merge(current_board, action)
        if np.all(current_board == board_after_merge):
            info = {'useless_action': True}
            return [current_board], info
        free_indices = np.argwhere(board_after_merge == 0)

        pop_up_number = 2  # TODO: replace this by random choice between 2 and 4
        next_states = []
        for index in free_indices:
            next_board = np.copy(board_after_merge)
            next_board[index] = pop_up_number
            next_states.append(next_board)
        info = {'useless_action': False}
        return next_states, info

    def action_reward(self, action):
        """
        Returns the reward after taking an action.
        """
        current_board = np.copy(self.board)
        board_after_merge = self.slide_and_merge(current_board, action)
        reward = self.calculate_reward(current_board, board_after_merge)
        return reward
