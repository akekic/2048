import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Gym2048-v0',
    entry_point='gym_2048.envs:Gym2048Env',
    nondeterministic=True,
)
