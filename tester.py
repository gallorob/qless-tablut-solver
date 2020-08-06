import csv
import os
from typing import Tuple, List

import gym
from gym import logger
from gym.wrappers import Monitor

from agent import Agent
from globals import records_dir, videos_dir, SETTINGS


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


def update_summary(winners: List[str], moves: List[int], epoch: int):
    # check if file exist already; if not, add the header for when we write
    has_header = os.path.isfile(os.path.join(records_dir, 'summary.csv'))
    # write summary
    with open(os.path.join(records_dir, 'summary.csv'), 'a+', newline='') as csvfile:
        fieldnames = ['epoch', 'def_wins', 'atk_wins', 'draws', 'max_moves', 'min_moves', 'avg_moves']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        def_wins = winners.count('DEF')
        atk_wins = winners.count('ATK')
        draws = len(winners) - def_wins - atk_wins
        max_moves = max(moves)
        min_moves = min(moves)
        avg_moves = min_moves + (max_moves - min_moves) // 2
        writer.writerow({
            'epoch': epoch,
            'def_wins': def_wins,
            'atk_wins': atk_wins,
            'draws': draws,
            'max_moves': max_moves,
            'min_moves': min_moves,
            'avg_moves': avg_moves
        })


def run_test(env: gym.Env, agents: Tuple[Agent, Agent], epoch: int) -> List[str]:
    """
    Run N matches as test

    :param env: The gym.Env
    :param agents: The two agents (Defender, Attacker)
    :param epoch: The current training epoch the agents are at
    """
    logger.info(f"Starting {SETTINGS.TEST_MATCHES} test match(es)")
    logger.info(f"video recording is {'enabled' if SETTINGS.RECORD_TEST_MATCHES else 'disabled'}")
    test_env = env if not SETTINGS.RECORD_TEST_MATCHES else Monitor(env=env,
                                                                    directory=os.path.join(videos_dir, str(epoch)),
                                                                    video_callable=lambda episode_id: (
                                                                                                                  episode_id + 1) % SETTINGS.TEST_MATCHES_RECORD_INTERVAL == 0)
    winners = []
    moves_len = []
    for ep in range(SETTINGS.TEST_MATCHES):
        moves = []
        obs = test_env.reset()
        if SETTINGS.RENDER_TEST_MATCHES:
            test_env.render()
        curr_agent = 0
        while True:
            action, _ = agents[curr_agent].choose_action(obs,
                                                         test_env.env.action_space if SETTINGS.RECORD_TEST_MATCHES else env.action_space)
            moves.append(test_env.env.actions[action] if SETTINGS.RECORD_TEST_MATCHES else env.actions[action])
            obs, _, done, info = test_env.step(action)
            if SETTINGS.RENDER_TEST_MATCHES:
                test_env.render()
            captures = info.get('captured')
            if len(captures) > 0:
                moves[-1] += 'x' + 'x'.join(captures)
            if done:
                write_match_infos(info, moves, f'match_{epoch}_{ep}')
                winners.append(info.get('winner', None))
                moves_len.append(len(moves))
                break
            curr_agent = 0 if curr_agent == 1 else 1
    test_env.close()
    update_summary(winners, moves_len, epoch)
    logger.info('Test match(es) completed and results saved')
    return winners
