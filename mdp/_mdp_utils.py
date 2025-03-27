import numpy as np


def get_random_policy(mdp, seed=None, deterministic=True):
    """
        :param mdp: the MDP object
        :param seed: the seed to control the randomness of the policy
        :return: a random policy for the MDP
    """
    rs = np.random.RandomState(seed)

    choices = {}

    def choose(s):
        if s not in choices:
            actions = mdp.get_actions_in_state(s)
            if not actions:
                raise ValueError(f"No action can be picked in a terminal state.")
            action = actions[rs.choice(range(len(actions)))]
            if deterministic:
                choices[s] = action
            else:
                return action
        return choices[s]

    return choose

def get_policy_from_dict(action_map):
    """
        :param action_map: keys are states, values are actions
        :return: a function f(s) that returns the action dictated by the table for the state s
    """
    return lambda s: action_map[s]
def get_closed_form_of_mdp(mdp):

    """
    :param mdp: the MDP object
    :return: triple (states, probs, rewards), where `states` and `rewards` are lists, and probs[s][a][s'] = P(s'|s,a)
    """
    states = list(mdp.states)
    probs = {}
    for s in states:
        p_s = {
            a: mdp.get_transition_distribution(s, a) for a in mdp.get_actions_in_state(s)
        }
        if p_s:
            probs[s] = p_s
    rewards = np.array([mdp.get_reward(s) for s in states])
    return states, probs, rewards
