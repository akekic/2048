from gym_2048.envs import Gym2048Env


def main():
    g = Gym2048Env(verbose=True)  # initialize board
    g.step(0)
    g.step(1)
    g.step(2)
    g.step(3)


main()
