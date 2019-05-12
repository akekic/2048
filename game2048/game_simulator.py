import numpy as np

from game2048.game import Game


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
