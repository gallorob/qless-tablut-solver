import os
from typing import Tuple, Optional, List

import gym
from gym import logger
from gym.wrappers import Monitor

from agent import Agent
from globals import records_dir


def write_match_infos(infos: dict, moves: List[str], name: str):
    """
    Save a small summary of the match in text file

    :param infos: The endgame dictionary
    :param moves: The list of all moves in string form
    :param name: The name for the match
    """
    with open(os.path.join(records_dir, name + '.match'), 'w') as f:
        f.write(f"Winner: {infos.get('winner', 'ATK/DEF')}\n")
        f.write(f"Reason: {infos.get('reason', 'NOT AVAILABLE')}\n")
        f.write('Moves:\n')
        for i in range(0, len(moves), 2):
            if i + 1 == len(moves):
                f.write(f'{i // 2 + 1}: {moves[i]}\n')
            else:
                f.write(f'{i // 2 + 1}: {moves[i]}  -  {moves[i + 1]}\n')


def run_test(env: gym.Env, agents: Tuple[Agent, Agent], epoch: int, n: int, record: Optional[bool] = False):
    """
    Run N matches as test

    :param env: The gym.Env
    :param agents: The two agents (Defender, Attacker)
    :param epoch: The current training epoch the agents are at
    :param n: The number of matches to run
    :param record: If True, wrap the environment in a Monitor and save the matches as videos as well
    """
    logger.info(f"Starting {n} test match(es); video recording is {'enabled' if record else 'disabled'}")
    test_env = env if not record else Monitor(env, os.path.join(records_dir, 'videos'))
    for ep in range(n):
        moves = []
        obs = env.reset()
        env.render()
        curr_agent = 0
        while True:
            action, _ = agents[curr_agent].choose_action(obs,
                                                         env.env.action_space if record else env.action_space)
            moves.append(env.env.actions[action] if record else env.actions[action])
            obs, _, done, info = test_env.step(action)
            test_env.render()
            captures = info.get('captured')
            if len(captures) > 0:
                moves[-1] += 'x' + 'x'.join(captures)
            if done:
                write_match_infos(info, moves, f'match_{epoch}_{ep}')
                break
            curr_agent = 0 if curr_agent == 1 else 1
    test_env.close()
    logger.info('Test match(es) completed and results saved')
