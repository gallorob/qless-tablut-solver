import gym

from agent import Agent
from trainer import train_loop

if __name__ == '__main__':
    env = gym.make('gym_tablut:Tablut-v0')
    defender = Agent()
    attacker = Agent()

    train_loop(env, (defender, attacker))
