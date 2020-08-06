from typing import List, Tuple, Optional
import pickle
import os

from globals import ATK, DEF


class Match:
    """
    A class that represents a match
    """
    def __init__(self, states: List[List[float]], moves: List[float], winner: Optional[str]):
        self.states = states
        self.moves = moves
        self.is_draw = True if winner is None else False
        self.winner = winner


class MatchesCollection:
    """
    A class that represents a collection of matches
    """
    def __init__(self, n_matches: int, train_epoch: int):
        self.name = f'matches_{train_epoch}_{n_matches}'
        self.n_matches = n_matches
        self.matches = {
            'ATK': [],
            'DEF': []
        }
        self.filled = False

    def get_info(self):
        """
        Get matches infos
        :return: The matches infos
        """
        infos = f'Matches info:\n\tMatches Collection for {self.n_matches} matches per player.'
        # atk infos
        atks = [len(atk[0]) for atk in self.matches.get('ATK')]
        defs = [len(def_[0]) for def_ in self.matches.get('DEF')]
        infos += f"\n\tAttacker:\n\tLongest match: {max(atks)} moves\n\t"
        infos += f"\tShortest match: {min(atks)} moves\n\t"
        infos += f"\tAverage length: {min(atks) + ((max(atks) - min(atks)) // 2)} moves"
        infos += f"\n\tDefender:\n\tLongest match: {max(defs)} moves\n\t"
        infos += f"\tShortest match: {min(defs)} moves\n\t"
        infos += f"\tAverage length: {min(defs) + ((max(defs) - min(defs)) // 2)} moves"
        return infos

    def shortest_match(self, player: str) -> int:
        """
        Compute the shortest match played

        :param player: The player
        :return: The number of moves of the shortest match
        """
        assert player in self.matches.keys()
        ns = [len(m[0]) for m in self.matches.get(player, None)]
        return min(ns)

    def longest_match(self, player: str) -> int:
        """
        Compute the longest match played

        :param player: The player
        :return: The number of moves of the longest match
        """
        assert player in self.matches.keys()
        ns = [len(m[0]) for m in self.matches.get(player, None)]
        return max(ns)

    def process_match(self, match: Match):
        """
        Process a completed match, updating the self.matches dictionary

        :param match: The completed match
        """
        # handle draws
        if match.is_draw:
            self.matches.get('DEF').append(self.generate_samples(match, 0))
            self.matches.get('ATK').append(self.generate_samples(match, 1))
        # handle wins
        elif len(self.matches.get(match.winner)) < self.n_matches:
            self.matches.get(match.winner).append(self.generate_samples(match, 0 if match.winner == 'DEF' else 1))
        # check if filled
        self.filled = len(self.matches.get('ATK')) == self.n_matches and len(self.matches.get('DEF')) == self.n_matches

    @staticmethod
    def generate_samples(match: Match, player: int) -> Tuple[List[List[float]], List[float]]:
        """
        Generate the samples according to the winning player

        :param match: The completed match
        :param player: The winning player
        :return: A tuple with (player_states_before, player_moves)
        """
        samples = []
        labels = []
        for i, (sample, label) in enumerate(zip(match.states, match.moves)):
            if i % 2 == (0 if player == DEF else 1):
                samples.append(sample)
                labels.append(label)
        return samples, labels


def save_matches(matches: MatchesCollection, path: Optional[str] = './'):
    """
    Save the matches as a .pkl file

    :param matches: The matches
    :param path: The path
    """
    with open(os.path.join(path, f'{matches.name}.pkl'), 'wb') as f:
        pickle.dump(matches, f)


def load_matches(name: str, path: Optional[str] = './') -> MatchesCollection:
    """
    Load the matches from a .pkl file

    :param name: The file name
    :param path: The path
    """
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

