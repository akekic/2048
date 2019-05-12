from game2048 import Game, GameSimulator, GymEnv2048


def main():
    g = GymEnv2048(verbose=True)  # initialize board
    print(g)  # print board and score
    g.step(0)
    g.step(1)
    g.step(2)
    g.step(3)


main()
# sim = GameSimulator(n_runs=10)
# sim.simulate_strategy(strategy='swirl')
# sim.summary()
