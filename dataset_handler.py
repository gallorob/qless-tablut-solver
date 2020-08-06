import os
import pickle
from typing import Optional

import numpy as np
import torch as th
import torchvision.transforms as transforms
from gym import logger
from torch.utils.data import Dataset

from globals import LAST_MOVES, SHAPE_STATE, datasets_dir
from match_handler import MatchesCollection


class TablutDataset(Dataset):
    def __init__(self, samples: np.array, labels: np.array, name: str, transform: Optional = None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
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
                with_limit = True
            else:
                n_samples = 0
                for m in ms:
                    n_samples += len(m[0])
                samples = np.empty(shape=(n_samples, SHAPE_STATE[0], SHAPE_STATE[1], SHAPE_STATE[2]),
                                   dtype=np.float)
                labels = np.empty(shape=n_samples)
                with_limit = False
            i = 0
            for m in ms:
                for s, l in zip(m[0][-(lm if with_limit else 0):], m[1][-(lm if with_limit else 0):]):
                    samples[i] = s
                    labels[i] = l
                    i += 1
            dataset = TablutDataset(samples=samples,
                                    labels=labels,
                                    name=f"{player}_{epoch}_{matches.n_matches}_{(lm if with_limit else 'full')}",
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
            save_dataset(dataset, datasets_dir)
    logger.info('All datasets built')


def save_dataset(dataset: TablutDataset, path: Optional[str] = './'):
    """
    Save the dataset as a .pkl file

    :param dataset: The dataset
    :param path: The path
    """
    with open(os.path.join(path, f'{dataset.name}.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


def load_dataset(epoch: int, limit: int, n_matches: int, player: str, path: Optional[str] = './') -> TablutDataset:
    """
    Load the dataset from a .pkl file

    :param epoch: The current epoch
    :param limit: The moves limit ('full' handled internally)
    :param n_matches: The number of matches
    :param player: The player
    :param path: The path
    """
    name = f"{player}_{epoch}_{n_matches}_{limit}.pkl"
    if not os.path.isfile(os.path.join(path, name)):
        name = f"{player}_{epoch}_{n_matches}_{'full'}.pkl"
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)
