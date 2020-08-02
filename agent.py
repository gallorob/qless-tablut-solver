from typing import List, Tuple

import numpy as np


class Agent():
    def __init__(self):
        self.net = None

    def train(self, data: Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], Tuple[np.array, np.array]]):
        """
        Train the agent

        :param data: A tuple with (train_data, val_data, test_data)
        """
        ((x_train, y_train), (x_val, y_val), (x_test, y_test)) = data

    def choose_action(self, curr_state: List[List[float]], possible_actions) -> Tuple[int, float]:
        """
        Choose the action given the current state

        :param curr_state: The current game state
        :param possible_actions: All valid actions
        :return: The action and the normalized action
        """
        if self.net is None:
            action = possible_actions.sample()
        else:
            action = self.net(curr_state)
        return action, (action / possible_actions.n)
