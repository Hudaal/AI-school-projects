import numpy as np
import random

class Actor():
    """The actor class in actor-critic system"""
    def __init__(self, lr, discount_f, decay_f, epsilon_greedy, epsilon_greedy_decay):
        self.lr = lr # learning rate
        self.discount_f = discount_f
        self.decay_f = decay_f
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_greedy_decay = epsilon_greedy_decay
        self.target_policy = {}
        self.behavior_policy = {}
        self.e = {}

    def init_policy(self, sim_world, all_states):
        # make start policy of all the states from the sim world
        for state in all_states:
            for action in sim_world.get_all_actions():
                self.target_policy[(state, action)] = 0

    def init_elig(self, sim_world, all_states):
        # make start eligability of all the states from the sim world
        for state in all_states:
            for action in sim_world.get_all_actions():
                self.e[(state, action)] = 0

    def get_best_policy(self, state, sim_world, all_states):
        # Chooses the best action by checking the best value in all possible actions
        all_possible_actions = sim_world.get_all_possible_actions()
        if random.random() < self.epsilon_greedy:
            return random.choice(all_possible_actions)
        best_action = all_possible_actions[random.randint(0, len(all_possible_actions)-1)]
        policy_value = -1000
        for action in all_possible_actions:
            if (state, action) not in self.target_policy.keys():
                self.target_policy[(state, action)] = policy_value
            V_s = self.target_policy[(state, action)]
            if V_s > policy_value:
                best_action = action
                policy_value = V_s
        # self.target_policy[(state, best_action)] = 0
        return best_action

    def update_target_policy(self, state, action, delta):
        # updates the target policy with the new reward and error
        if (state, action) not in self.target_policy.keys():
            self.target_policy[(state, action)] = -1000
        if (state, action) not in self.e.keys():
            self.e[(state, action)] = 1
        if np.array(delta).shape:
            delta_sum = self.sum_all(np.array(delta))
        else:
            delta_sum = delta
        self.target_policy[(state, action)] += self.lr * delta_sum * self.e[(state, action)]

    def update_eligibility(self, state, action):
        # updates eligability
        if (state, action) not in self.e.keys():
            self.e[(state, action)] = 1
        self.e[(state, action)] *= self.discount_f * self.decay_f

    def sum_all(self, array):
        # this used if the NN critic is used to sum up the target from it
        sumall = 0
        for item in array:
            sumall += item
        return sumall
