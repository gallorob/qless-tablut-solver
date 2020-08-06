import gym
from torch.utils.tensorboard import SummaryWriter

from agent import *
from dataset_handler import *
from globals import matches_dir, SETTINGS, custom_games
from match_handler import *
from tester import run_test


def train_loop(env: gym.Env, agents: Tuple[Agent, Agent]):
    """
    Training loop for the agents

    :param env: The gym.Env
    :param agents: A tuple with the two agents: (defender, attacker)
    """
    writer = SummaryWriter(log_dir=f'runs/OuterLoopTrain')

    logger.info(f'Starting training...')
    for epoch in range(SETTINGS.EPOCHS):
        logger.info(f'Epoch {epoch + 1}/{SETTINGS.EPOCHS}:')
        if SETTINGS.SIMULATE_MATCHES:
            simulated = simulate_matches(env, agents, SETTINGS.N_MATCHES, epoch, SETTINGS.USE_CUSTOM_GAMES)
        else:
            sim_name = f'matches_{epoch}_{SETTINGS.N_MATCHES}.pkl'
            simulated = load_matches(sim_name, matches_dir)
        if SETTINGS.GENERATE_DATASET:
            build_dataset(simulated, epoch + 1)
        for player, agent in zip(['DEF', 'ATK'], agents):
            logger.info(f'Training {player} agent...')
            for limit in LAST_MOVES:
                logger.info(f'Training with at most last {limit} moves...')
                dataset = load_dataset(epoch + 1, limit, SETTINGS.N_MATCHES, player, datasets_dir)
                agent.train(dataset, limit)
            logger.info(f'Agent {player} trained.')
        winners = run_test(env, agents, epoch + 1)
        writer.add_scalar(f"AgentWins/DEF", sum(1 for i in winners if i == 'DEF'), epoch)
        writer.add_scalar(f"AgentWins/ATK", sum(1 for i in winners if i == 'ATK'), epoch)
        writer.add_scalar(f"AgentWins/DRAWS", sum(1 for i in winners if i is None), epoch)
        writer.flush()
    writer.close()


def simulate_matches(env: gym.Env, agents: Tuple[Agent, Agent], n_matches: int, epoch: int,
                     use_custom: bool) -> MatchesCollection:
    """
    Simulate N matches with the current agents

    :param env: The gym.Env
    :param agents: A tuple with the two agents: (defender, attacker)
    :param n_matches: The number of matches per player to simulate at each epoch
    :param epoch: The current epoch of training
    :param use_custom: If true, uses the custom games
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
        step = 0
        if SETTINGS.RENDER_SIMULATED_MATCHES:
            env.render()
        while True:
            if use_custom:
                action = custom_games.get(epoch)[sim_counter][step]
                action = np.where(env.actions == action)[0][0]
                move = action / env.action_space.n
            else:
                action, move = agents[curr_agent].choose_action(states[len(states) - 1], env.action_space)
            moves.append(move)
            obs, _, done, info = env.step(action)
            if SETTINGS.RENDER_SIMULATED_MATCHES:
                env.render()
            states.append(obs)
            if done:
                match = Match(states, moves, info.get('winner', None))
                matches.process_match(match)
                logger.info(f"Current simulation status: DEF: {len(matches.matches.get('DEF'))}/{n_matches}"
                            f" ; ATK: {len(matches.matches.get('ATK'))}/{n_matches}")
                break
            curr_agent = 0 if curr_agent == 1 else 1
            step += 1
        sim_counter += 1
    logger.info(f'Simulated {sim_counter} matches before reaching {n_matches} matches per player')

    logger.debug(matches.get_info())
    save_matches(matches, matches_dir)
    logger.debug(f'Saved {matches.name}')
    return matches
