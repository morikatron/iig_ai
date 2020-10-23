from collections import deque
from itertools import combinations


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
        self.information = ((), ())  # (private card, history)

        self.pi = 0
        self.pi_mi = 0  # pi_-i
        self.pi_i = 0  # pi_i
        self.true_pi_mi = 0  # pi_-i following current average strategy profile
        self.eu = eu
        self.cv = 0
        self.cfr = {}  # counter-factual regret of not taking action a at history h(not information I)
        self.pi_i_sum = 0  # denominator of average strategy
        self.pi_sigma_sum = {}  # numerator of average strategy
        self.num_updates = 0

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

    def get_nash_equilibrium(self, node: Node, strategy_profile=None):
        """
        nash equilibrium (alpha = 0)
        """
        if node.terminal:
            return strategy_profile
        if strategy_profile is None:
            strategy_profile = {player: {} for player in range(-1, self.num_players)}
        if node.information not in strategy_profile[node.player]:
            hand = node.information[0]
            if len(hand) == 0:
                strategy_profile[node.player][node.information] = {action: 1 / len(node.children) for action in node.children}
            else:
                card = hand[0]
                for action, child in node.children.items():
                    if node.player == 0:
                        if action == "bet":
                            p = 0
                        elif action == "check":
                            p = 1
                        elif action == "call":
                            if card == 0:
                                p = 0
                            elif card == 1:
                                p = 1/3
                            else:
                                p = 1
                        else:
                            if card == 0:
                                p = 1
                            elif card == 1:
                                p = 2/3
                            else:
                                p = 0
                    else:
                        if action == "bet":
                            if card == 0:
                                p = 1/3
                            elif card == 1:
                                p = 0
                            else:
                                p = 1
                        elif action == "check":
                            if card == 0:
                                p = 2/3
                            elif card == 1:
                                p = 1
                            else:
                                p = 0
                        elif action == "call":
                            if card == 0:
                                p = 0
                            elif card == 1:
                                p = 1/3
                            else:
                                p = 1
                        else:
                            if card == 0:
                                p = 1
                            elif card == 1:
                                p = 2/3
                            else:
                                p = 0
                    if node.information not in strategy_profile[node.player]:
                        strategy_profile[node.player][node.information] = {}
                    strategy_profile[node.player][node.information][action] = p
        for child in node.children.values():
            strategy_profile = self.get_nash_equilibrium(child, strategy_profile=strategy_profile)
        return strategy_profile


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

