from typing import Optional, Tuple, Any, Union
from gym import logger
from match_handler import MatchesCollection
from globals import LAST_MOVES, SHAPE_STATE, datasets_dir, TORCH_SHAPE_STATE
from torch.utils.data import Dataset
import torch as th
import os

import numpy as np

from globals import SETTINGS


class TablutDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]
        sample = sample.reshape(TORCH_SHAPE_STATE)
        label = self.labels[idx]
        return sample, label


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


def load_dataset(epoch: int, limit: int, n_matches: int, player: str) -> TablutDataset:
    """
    Load the dataset

    :param epoch: The current epoch
    :param limit: The moves limit
    :param n_matches: The number of matches per player
    :param player: The current player
    :return: The processed train, validation and testing arrays
    """
    s_name = f'{player}_states_{epoch}_{n_matches}_{limit}.dat'
    l_name = f'{player}_labels_{epoch}_{n_matches}_{limit}.dat'
    if os.path.isfile(os.path.join(datasets_dir, s_name)):
        samples = load_data(name=s_name,
                            shape=(n_matches * limit, SHAPE_STATE[0], SHAPE_STATE[1], SHAPE_STATE[2]),
                            path=datasets_dir)
        labels = load_data(name=l_name,
                           shape=(n_matches * limit),
                           path=datasets_dir)
    else:
        s_name = f'{player}_states_{epoch}_{n_matches}_full.desc'
        l_name = f'{player}_labels_{epoch}_{n_matches}_full.desc'
        samples = load_from_descriptor(name=s_name,
                                       path=datasets_dir)
        labels = load_from_descriptor(name=l_name,
                                      path=datasets_dir)
    return prepare_dataset(samples, labels)


def prepare_dataset(samples: np.array, labels: np.array) -> TablutDataset:
    """
    Shuffle and split the samples and labels

    :param samples: The samples
    :param labels: The labels
    :return: The processed train, validation and testing arrays
    """
    dataset = TablutDataset(th.from_numpy(samples), th.from_numpy(labels))
    return dataset


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


def load_data(name: str, shape: Union[Any, Tuple[int, int, int, int]], path: Optional[str] = './') -> np.array:
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
