from typing import Optional, Tuple
from gym import logger
from match_handler import MatchesCollection
from globals import LAST_MOVES, SHAPE_STATE, datasets_dir

import os

import numpy as np


def build_dataset(matches: MatchesCollection, epoch: int):
    """
    Build the dataset for both Attacker and Defender and at the various last-moves windows

    :param matches: The matches as a collection
    :param epoch: The current epoch for training
    """
    logger.info('Starting building dataset...')
    for player in ['ATK', 'DEF']:
        logger.info(f'Building dataset for {player}...')
        ms = matches.matches.get(player)
        for lm in LAST_MOVES:
            if lm <= matches.shortest_match(player):
                samples = np.empty(shape=(matches.n_matches * lm, SHAPE_STATE[0], SHAPE_STATE[1], SHAPE_STATE[2]),
                                   dtype=np.float)
                labels = np.empty(shape=(matches.n_matches * lm))
                i = 0
                for m in ms:
                    for s, l in zip(m[0][-lm:], m[1][-lm:]):
                        samples[i] = s
                        labels[i] = l
                        i += 1
                # save on disk
                s_name = f'{player}_states_{epoch}_{matches.n_matches}_{lm}.dat'
                save_data(samples, s_name, datasets_dir)
                l_name = f'{player}_labels_{epoch}_{matches.n_matches}_{lm}.dat'
                save_data(labels, l_name, datasets_dir)
            else:
                n_samples = 0
                for m in ms:
                    n_samples += len(m[0])
                samples = np.empty(shape=(n_samples, SHAPE_STATE[0], SHAPE_STATE[1], SHAPE_STATE[2]),
                                   dtype=np.float)
                labels = np.empty(shape=n_samples)
                i = 0
                for m in ms:
                    for s, l in zip(m[0], m[1]):
                        samples[i] = s
                        labels[i] = l
                        i += 1
                # save on disk
                s_name = f'{player}_states_{epoch}_{matches.n_matches}_full.dat'
                save_data(samples, s_name, datasets_dir, True)
                l_name = f'{player}_labels_{epoch}_{matches.n_matches}_full.dat'
                save_data(labels, l_name, datasets_dir, True)
    logger.info('All datasets built')


def save_data(data: np.array, name: str, path: Optional[str] = './', descriptor: bool = False):
    """
    Save the data as memory map on disk as a .dat file

    :param data: The data array
    :param name: The name of the data
    :param path: The path
    :param descriptor: If True, generate also a descriptor with the data shape
    """
    # create memmap
    filename = os.path.join(path, name)
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    logger.debug(f'Saved dataset {name}')
    if descriptor:
        with open(os.path.join(path, f'{name[:-4]}.desc'), 'w') as f:
            f.write(str(data.shape))
    # flush to disk
    del fp


def load_data(name: str, shape: Tuple[int, int, int, int], path: Optional[str] = './') -> np.array:
    """
    Load the dataset from the memory map .dat file

    :param name: The name of the .dat file
    :param shape: The memory map shape
    :param path: The path
    :return: The memory map with the data
    """
    filename = os.path.join(path, name)
    mmap = np.memmap(filename, dtype='float32', mode='r', shape=shape)
    return mmap


def load_from_descriptor(name: str, path: Optional[str] = './') -> np.array:
    """
    Load the dataset from the descriptor file (used for `full` matches)

    :param name: The descriptor file name
    :param path: The path
    :return: The memory map with the data
    """
    with open(os.path.join(path, name), 'r') as f:
        shape = eval(f.readline())
    return load_data(name[:-5]+'.dat', shape, path)
