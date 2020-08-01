import gym

from agent import *
from dataset_handler import *
from globals import matches_dir
from match_handler import *

from tester import run_test

# import argparse

# set the level for the logger
logger.set_level(logger.INFO)


def train_loop(env: gym.Env, agents: Tuple[Agent, Agent], n_matches: int, epochs: int, use_existing: bool = False):
    """
    Training loop for the agents

    :param env: The gym.Env
    :param agents: A tuple with the two agents: (defender, attacker)
    :param n_matches: The number of matches per player to simulate at each epoch
    :param epochs: The number of epochs to train the agents for
    :param use_existing: If true, load the simulated matches instead of simulating them
    """
    logger.info(f'Starting training...')
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}:')
        if not use_existing:
            simulated = simulate_matches(env, agents, n_matches, epoch)
        else:
            sim_name = f'matches_{epoch}_{n_matches}.pkl'
            simulated = load_matches(sim_name, matches_dir)
        build_dataset(simulated, epochs)


def simulate_matches(env: gym.Env, agents: Tuple[Agent, Agent], n_matches: int, epoch: int) -> MatchesCollection:
    """
    Simulate N matches with the current agents

    :param env: The gym.Env
    :param agents: A tuple with the two agents: (defender, attacker)
    :param n_matches: The number of matches per player to simulate at each epoch
    :param epoch: The current epoch of training
    :return: A filled MatchesCollection object
    """
    matches = MatchesCollection(n_matches=n_matches,
                                train_epoch=epoch)
    sim_counter = 0
    while not matches.filled:
        states = []
        moves = []
        states.append(env.reset())
        curr_agent = 0
        while True:
            action, move = agents[curr_agent].choose_action(states[len(states) - 1], env.action_space)
            moves.append(move)
            obs, _, done, info = env.step(action)
            states.append(obs)
            if done:
                match = Match(states, moves, info.get('winner', None))
                matches.process_match(match)
                break
            curr_agent = 0 if curr_agent == 1 else 1
        sim_counter += 1
    logger.info(f'Simulated {sim_counter} matches before reaching {n_matches} matches per player')

    logger.debug(matches.get_info())
    save_matches(matches, matches_dir)
    logger.debug(f'Saved {matches.name}')
    return matches


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Solver for the TablutEnv')
    env = gym.make('gym_tablut:Tablut-v0')
    defender = Agent()
    attacker = Agent()
    n_matches = 20
    epochs = 1
    # use_existing = False
    # train_loop(env, (defender, attacker), n_matches, epochs, use_existing)

    test_matches = 4
    test_record = False
    run_test(env, (defender, attacker), epochs, test_matches, test_record)

    # name = 'DEF_labels_1_5_full.desc'
    # ls = load_from_descriptor(name, datasets_dir)
    # print(ls.shape)
    # print(ls)
