from game2048 import GymEnv2048


def main():
    g = GymEnv2048(verbose=True)  # initialize board
    g.step(0)
    g.step(1)
    g.step(2)
    g.step(3)


main()
