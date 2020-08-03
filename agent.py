from typing import List, Tuple

from dataset_handler import TablutDataset

from gym import logger
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader

from globals import SETTINGS


class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=9,
                               kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=9,
                               out_channels=15,
                               kernel_size=(3, 3))
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(batch_size*3*3*15, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 1*batch_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = th.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.relu(x)
        return output


class Agent():
    def __init__(self):
        self.has_trained = False
        self.epochs = 50
        self.batch_size = 1
        self.net = Net(self.batch_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)  

    def train(self, dataset: TablutDataset):
        """
        Train the agent

        :param data: A tuple with (train_data, val_data, test_data)
        """
        dataset_size = len(dataset)
        train_size = math.floor(SETTINGS.TRAIN_TEST_SPLIT * dataset_size)
        train_dataset, test_dataset = random_split(dataset, [train_size, dataset_size - train_size])
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 drop_last=True)

        for ep in range(self.epochs):
            logger.info(f'\tEpoch {ep + 1}/{self.epochs}')
            for phase in ['train', 'test']:
                self.net.train() if phase == 'train' else self.net.eval()
                running_loss = 0.0
                running_acc = 0.0
                for batch_index, (samples, labels) in enumerate(train_loader):
                    with th.set_grad_enabled(phase == 'train'):
                        self.optimizer.zero_grad()
                        out = self.net(samples)
                        loss = self.criterion(out, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * train_size
                    running_acc += th.sum(out.data == labels.data)  
                    if batch_index % 10 == 0:
                        logger.info(f'\t[{phase}: Loss: {running_loss / train_size}] Acc: {running_acc / train_size}')
                        running_loss = 0.0
        self.has_trained = True


    def choose_action(self, curr_state: List[List[float]], possible_actions) -> Tuple[int, float]:
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
            mod_state = curr_state.astype('float32')
            mod_state = mod_state.reshape((1, 3, 9, 9))
            mod_state = th.from_numpy(mod_state)
            action = self.net(mod_state).item()
            # action = actions[0]
            return int(action * possible_actions.n), action
