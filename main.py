import gym
from gym import logger
import torch as th
import numpy as np

from agent import Agent
from trainer import train_loop

if __name__ == '__main__':

    # set the level for the logger
    logger.set_level(logger.INFO)

    np.random.seed(0)
    th.manual_seed(0)

    env = gym.make('gym_tablut:Tablut-v0')
    defender = Agent()
    attacker = Agent()

    train_loop(env, (defender, attacker))
