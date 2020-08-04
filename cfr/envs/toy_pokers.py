from collections import deque
from itertools import combinations

import numpy as np


def add_list_to_dict(target_dict, key, value):
    if key in target_dict.keys():
        target_dict[key].append(value)
    else:
        target_dict[key] = [value]


class Node:
    def __init__(self, player, terminal, eu=0):
        self.children = {}
        self.player = player
        self.terminal = terminal
        self.private_cards = []
        self.history = []
        self.information = ()  # (private card, history)

        self.pi = 0
        self.pi_mi = 0  # pi_-i
        self.pi_i = 0  # pi_i
        self.eu = eu
        self.cv = 0
        self.cfr = {}  # counterfactual regret of not taking action a at history h(not information I)
        self.pi_i_sum = 0  # denominator of average strategy
        self.pi_sigma_sum = {}  # numerator of averate strategy

    def expand_child_node(self, action, next_player, terminal, utility=0, private_cards=None):
        next_node = Node(next_player, terminal, utility)
        self.children[action] = next_node
        self.cfr[action] = 0
        self.pi_sigma_sum[action] = 0
        next_node.private_cards = self.private_cards if private_cards is None else private_cards
        next_node.history = self.history + [action] if self.player != -1 else self.history
        next_node.information = (next_node.private_cards[next_player], tuple(next_node.history))
        return next_node


class Card:
    def __init__(self, rank, suit=None):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        if self.suit is None:
            return str(self.rank)
        else:
            return str(self.rank) + str(self.suit)


class KuhnPoker:
    def __init__(self):
        self.num_players = 2
        self.deck = [i for i in range(3)]
        self.information_sets = {player: {} for player in range(-1, self.num_players)}
        self.root = self._build_game_tree()

    def _build_game_tree(self):
        stack = deque()
        next_player = -1
        root = Node(next_player, False)
        add_list_to_dict(self.information_sets[next_player], root.information, root)
        for hand_0 in combinations(self.deck, 1):
            for hand_1 in combinations(self.deck, 1):
                if set(hand_0) & set(hand_1):
                    continue
                private_cards = [hand_0, hand_1, ()]  # p1, p2, chance player
                next_player = 0
                node = root.expand_child_node(str(*hand_0) + ',' + str(*hand_1), next_player, False, private_cards=private_cards)
                add_list_to_dict(self.information_sets[next_player], node.information, node)
                stack.append(node)
                for action in ['check', 'bet']:  # player 0 actions
                    next_player = 1
                    node = node.expand_child_node(action, next_player, False)
                    add_list_to_dict(self.information_sets[next_player], node.information, node)
                    stack.append(node)
                    if action == 'check':
                        for action in ['check', 'bet']:  # player 1 actions
                            if action == 'check':
                                utility = self._compute_utility(action, next_player, hand_0, hand_1)
                                next_player = -1
                                node = node.expand_child_node(action, next_player, True, utility)
                                add_list_to_dict(self.information_sets[next_player], node.information, node)
                                node = stack.pop()
                            if action == 'bet':
                                next_player = 0
                                node = node.expand_child_node(action, next_player, False)
                                add_list_to_dict(self.information_sets[next_player], node.information, node)
                                stack.append(node)
                                for action in ['fold', 'call']:  # player 0 actions
                                    utility = self._compute_utility(action, next_player, hand_0, hand_1)
                                    next_player = -1
                                    node = node.expand_child_node(action, next_player, True, utility)
                                    add_list_to_dict(self.information_sets[next_player], node.information, node)
                                    node = stack.pop()
                    if action == 'bet':
                        stack.append(node)
                        for action in ['fold', 'call']:  # player 1 actions
                            utility = self._compute_utility(action, next_player, hand_0, hand_1)
                            next_player = -1
                            node = node.expand_child_node(action, next_player, True, utility)
                            add_list_to_dict(self.information_sets[next_player], node.information, node)
                            node = stack.pop()
        return root

    def _compute_utility(self, action, player, hand_0, hand_1):
        card_0, card_1 = hand_0[0], hand_1[0]
        is_win = card_0 > card_1
        if action == "fold":
            utility = 1 if player == 1 else -1
        elif action == "check":
            utility = 1 if is_win else -1
        elif action == "call":
            utility = 2 if is_win else -2
        else:
            utility = 0
        return utility


if __name__ == "__main__":
    kuhn_poker = KuhnPoker()
