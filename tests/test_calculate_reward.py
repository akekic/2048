import pytest

from gym_2048.envs.gym_2048_env import Gym2048Env

board1 = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
board2 = [
    [0, 0, 2, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
board3 = [
    [0, 0, 0, 4],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
board4 = [
    [0, 0, 0, 2],
    [0, 0, 0, 2],
    [0, 0, 0, 2],
    [0, 0, 0, 2],
]
board5 = [
    [0, 0, 0, 4],
    [0, 0, 0, 4],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
board6 = [
    [2, 0, 0, 0],
    [2, 0, 0, 0],
    [2, 0, 0, 0],
    [2, 0, 0, 0],
]
board7 = [
    [8, 8, 8, 8],
    [8, 16, 16, 32],
    [256, 0, 0, 0],
    [2, 2, 32, 32],
]
board8 = [
    [0, 0, 16, 16],
    [0, 8, 32, 32],
    [0, 0, 0, 256],
    [0, 0, 4, 64],
]


def test_empty_board():
    env = Gym2048Env()
    reward = env.calculate_reward(board1, board1)
    assert reward == 0


def test_1():
    env = Gym2048Env()
    assert env.calculate_reward(board2, board3) == 4
    assert env.calculate_reward(board4, board5) == 8
    assert env.calculate_reward(board4, board6) == 0
    assert env.calculate_reward(board7, board8) == 132
