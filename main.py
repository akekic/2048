from game2048 import Game, GameSimulator


def main():
    g = Game()  # initialize board
    print(g)  # print board and score
    g.move_right()
    g.move_up()
    g.move_down()
    g.move_left()
    # g._random_moves(100000)


main()
sim = GameSimulator(n_runs=10)
sim.simulate_strategy(strategy='swirl')
sim.summary()
