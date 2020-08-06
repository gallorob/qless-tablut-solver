import math
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from gym import logger
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_handler import TablutDataset
from globals import SETTINGS


class Agent:
    def __init__(self):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.has_trained = False
        self.epochs = 200
        self.batch_size = 16
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3,
                                out_channels=32,
                                kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=(3, 3))),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=(3, 3))),
            ('relu3', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(128, 64)),
            ('relu4', nn.ReLU()),
            ('dropout1', nn.Dropout(0.25)),
            ('fc2', nn.Linear(64, 1)),
            ('relu5', nn.ReLU())
        ])).double().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9)
        self.epsilon = 0.99
        self.epsilon_dec = 0.01

    def train(self, dataset: TablutDataset, lm: int):
        """
        Train the agent

        :param dataset: The dataset
        :param lm: The limit of moves
        """
        writer = SummaryWriter(log_dir=f'runs/{dataset.name}_{lm}')

        dataset_size = len(dataset)
        train_size = math.floor(SETTINGS.TRAIN_TEST_SPLIT * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 drop_last=False)

        for ep in range(self.epochs):
            logger.info(f'\tEpoch {ep + 1}/{self.epochs}')
            for phase in ['train', 'test']:
                self.net.train() if phase == 'train' else self.net.eval()
                ref_size = (train_size if phase == 'train' else test_size)
                running_loss = 0.0
                running_acc = 0.0
                for batch_index, (samples, labels) in enumerate(train_loader if phase == 'train' else test_loader):
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    with th.set_grad_enabled(phase == 'train'):
                        self.optimizer.zero_grad()
                        out = self.net(samples).squeeze(1)
                        out = th.clamp(out, min=0, max=1)
                        loss = self.criterion(out, labels)
                        writer.add_scalar(f"Loss/{phase}", loss, ep)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item()
                    acc = np.sum(np.round(out.data, 3).numpy() == np.round(labels.data, 3).numpy())
                    writer.add_scalar(f"Accuracy/{phase}", acc, ep)
                    running_acc += acc
                logger.info(f"\t[{phase}: Loss: {running_loss / ref_size}] Acc: {running_acc / ref_size}")
        self.has_trained = True
        self.epsilon -= self.epsilon_dec
        writer.flush()
        writer.close()

    def choose_action(self, curr_state: np.array, possible_actions) -> Tuple[int, float]:
        """
        Choose the action given the current state

        :param curr_state: The current game state
        :param possible_actions: All valid actions
        :return: The action and the normalized action
        """
        if not self.has_trained:
            action = possible_actions.sample()
            return action, action / possible_actions.n
        else:
            if np.random.uniform(0, 1) > self.epsilon:
                action = possible_actions.sample()
                return action, action / possible_actions.n
            else:
                mod_state = np.moveaxis(curr_state, 2, 0)
                mod_state = th.from_numpy(mod_state)
                mod_state = mod_state.unsqueeze(0)
                action = self.net(mod_state).item()
                return int(action * possible_actions.n), action
