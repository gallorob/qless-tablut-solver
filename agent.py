from typing import List, Tuple


class Agent():
    def __init__(self):
        self.net = None

    def choose_action(self, curr_state: List[List[float]], possible_actions) -> Tuple[int, float]:
        if self.net is None:
            action = possible_actions.sample()
        else:
            action = self.net(curr_state)
        return action, (action / possible_actions.n)
