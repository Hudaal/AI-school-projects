import numpy as np

class Critic_table_based:
    """The table based critic"""
    def __init__(self, lr, disount_f, decay_f):
        self.lr = lr
        self.discount_f = disount_f
        self.decay_f = decay_f
        self.e = {}
        self.v = {}

    def init_values(self, sim_world, all_states):
        # start with small random value
        for state in all_states:
            self.v[state] = np.random.uniform(-1.e-10, 1.e-10)

    def init_elig(self, sim_world, all_states):
        # start all with 0
        for state in all_states:
            self.e[state] = 0

    def update_state_eligibility(self, state, delta):
        self.v[state] += self.lr * delta * self.e[state]
        self.e[state] *= self.discount_f * self.decay_f

    def get_state_value(self, state):
        if state not in self.v.keys():
            self.v[state] = np.random.uniform(-1.e-8, 1.e-8)
        return self.v[state]

