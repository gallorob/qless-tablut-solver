import gym

from agent import *
from dataset_handler import *
from globals import matches_dir, SETTINGS
from match_handler import *

from tester import run_test


# set the level for the logger
logger.set_level(logger.INFO)


def train_loop(env: gym.Env, agents: Tuple[Agent, Agent]):
    """
    Training loop for the agents

    :param env: The gym.Env
    :param agents: A tuple with the two agents: (defender, attacker)
    """
    logger.info(f'Starting training...')
    for epoch in range(SETTINGS.EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{SETTINGS.EPOCHS}:')
        if SETTINGS.SIMULATE_MATCHES:
            simulated = simulate_matches(env, agents, SETTINGS.N_MATCHES, epoch)
        else:
            sim_name = f'matches_{epoch}_{SETTINGS.N_MATCHES}.pkl'
            simulated = load_matches(sim_name, matches_dir)
        if SETTINGS.GENERATE_DATASET:
            build_dataset(simulated, epoch + 1)
        for player, agent in zip(['DEF', 'ATK'], agents):
            logger.info(f'Training {player} agent...')
            for limit in LAST_MOVES:
                (train, val, test) = load_dataset(epoch + 1, limit, SETTINGS.N_MATCHES, player)
                agent.train((train, val, test))
            logger.info(f'Agent {player} trained.')
        run_test(env, agents, epoch + 1)


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
