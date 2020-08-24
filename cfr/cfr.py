from copy import copy
from copy import deepcopy
import yaml

from tqdm import tqdm

from envs.toy_pokers import Node, KuhnPoker


def update_pi(node: Node, policy: dict, pi_mi: list, pi_i: list):
    """
    update probability
    """
    node.pi = pi_mi[node.player] * pi_i[node.player]
    node.pi_mi = pi_mi[node.player]
    node.pi_i = pi_i[node.player]

    if node.terminal:
        return
    for action, child_node in node.children.items():
        next_pi_mi = copy(pi_mi)
        next_pi_i = copy(pi_i)
        for player in policy.keys():
            if player == node.player:
                next_pi_i[player] *= policy[node.player][node.information][action]
            else:
                next_pi_mi[player] *= policy[node.player][node.information][action]
        update_pi(child_node, policy, next_pi_mi, next_pi_i)


def update_node_values(node: Node, strategy_profile: dict):
    node_eu = 0  # ノードに到達した後得られる期待利得
    if node.terminal:
        return node.eu
    node.pi_i_sum += node.pi_i
    for action, child_node in node.children.items():
        p = strategy_profile[node.player][node.information][action]
        node.pi_sigma_sum[action] += node.pi_i * p
        node_eu += p * update_node_values(child_node, strategy_profile)
    node.eu = node_eu
    node.cv = node.pi_mi * node_eu
    for action, child_node in node.children.items():
        cfr = node.pi_mi * child_node.eu - node.cv if node.player == 0 else (-1) * (node.pi_mi * child_node.eu - node.cv)
        node.cfr[action] += cfr
    return node_eu


def get_initial_strategy(node: Node, num_players=None, strategy_profile=None):
    if node.terminal:
        return strategy_profile
    if strategy_profile is None:
        strategy_profile = {player: {} for player in range(-1, num_players)}  # chance nodeのために+1する
    if node.information not in strategy_profile[node.player]:
        strategy_profile[node.player][node.information] = {action: 1 / len(node.children) for action in node.children}
    for child in node.children.values():
        strategy_profile = get_initial_strategy(child, strategy_profile=strategy_profile)
    return strategy_profile


def update_strategy(strategy_profile: dict, average_strategy_profile: dict, information_sets: dict):
    for player, information_policy in strategy_profile.items():
        if player == -1:
            continue
        for information, strategy in information_policy.items():
            cfr = {}
            for action in strategy.keys():
                cfr_in_info = 0
                average_strategy_denominator = 0
                average_strategy_numerator = 0
                for same_info_node in information_sets[player][information]:
                    cfr_in_info += same_info_node.cfr[action]
                    average_strategy_denominator += same_info_node.pi_i_sum
                    average_strategy_numerator += same_info_node.pi_sigma_sum[action]
                cfr[action] = max(cfr_in_info, 0)
                average_strategy_profile[player][information][action] = average_strategy_numerator / average_strategy_denominator
            cfr_sum = sum([cfr_values for cfr_values in cfr.values()])
            for action in strategy.keys():
                if cfr_sum > 0:
                    strategy[action] = cfr[action] / cfr_sum
                else:
                    strategy[action] = 1 / len(strategy)
    return


def train(num_iter):
    kuhn_poker = KuhnPoker()
    strategy_profile = get_initial_strategy(kuhn_poker.root, kuhn_poker.num_players)
    average_strategy_profile = deepcopy(strategy_profile)

    for _ in tqdm(range(num_iter)):
        update_pi(kuhn_poker.root, strategy_profile, [1.0 for _ in range(kuhn_poker.num_players + 1)], [1.0 for _ in range(kuhn_poker.num_players + 1)])
        update_node_values(kuhn_poker.root, strategy_profile)
        update_strategy(strategy_profile, average_strategy_profile, kuhn_poker.information_sets)

    return average_strategy_profile


def main():
    average_strategy_profile = train(1000000)
    result = {}
    for player, sigma in average_strategy_profile.items():
        if player == -1:
            continue
        for info, p_dist in sigma.items():
            add_dtd(result, info[0][0])
            if len(info[1]) == 0:
                history = "-"
            else:
                history = "-".join(info[1])
            for action, p in p_dist.items():
                add_dtd(result[info[0][0]], history)
                result[info[0][0]][history][action] = p
    with open("../sample_result.yaml", "w") as f:
        yaml.dump(result, f)
    return


def add_dtd(d: dict, key):
    if key not in d:
        d[key] = {}


if __name__ == "__main__":
    main()
