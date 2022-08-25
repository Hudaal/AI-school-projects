from turtle import position
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

class Gambler_SimWorld:
    def __init__(self, prob=0.5, maxUnits=100, display=False) -> None:
        self.all_time_steps = []
        self.prob = prob
        self.maxUnits = maxUnits
        self.display = display
        self.reset()
        self.state_best = []
        for state in self.get_all_states():
            self.state_best.append([state, 0])

    def reset(self):
        # start with random units number gained
        self.units_gained = np.random.randint(1, self.maxUnits-1)
        self.terminated = False
        self.lost = 0
        self.won_lose_prob_list = []
        self.time_steps = 0

    def get_init_state(self):
        return self.units_gained

    def get_all_possible_actions(self):
        units_needed = self.maxUnits - self.units_gained
        if self.display:
            print('self.units_gained', self.units_gained, '  units_needed', units_needed)
        # all_possible_actions = list(range(1, units_needed + 1))
        # Every unit number until get the target units
        all_possible_actions = [i for i in range(1, units_needed + 1) if i <= self.units_gained]
        if len(all_possible_actions) == 0:
            return [0]
        return all_possible_actions

    def get_all_actions(self):
        return list(range(1, self.maxUnits))

    def do_action(self, units_chosen):
        # the action is the units to add
        if units_chosen > self.state_best[self.units_gained][1]:
            self.state_best[self.units_gained][1] = units_chosen # To find the higher units chosen for each state

        won_lose_prob = np.random.uniform(0, 1) # random probability and combine it with the probability we have
        self.won_lose_prob_list.append(won_lose_prob)
        if self.display:
            print('won_lose_prob', won_lose_prob)
        reward = 0
        if won_lose_prob <= self.prob: # win this part
            self.units_gained += units_chosen
            reward += 5
        else: # lose this part
            self.units_gained -= units_chosen
            reward -= 4
        self.time_steps += 1

        if self.units_gained == self.maxUnits:
            reward += 22 # won the game
            self.terminated = True
        elif self.units_gained <= 0:
            reward -= 20 # lost the game
            self.terminated = True
        if self.terminated:
            if self.time_steps <= 2:
                reward -= 5
            elif self.time_steps <= 3:
                reward -= 4
            elif self.time_steps <= 4:
                reward -= 3
            elif self.time_steps <= 5:
                reward -= 2
            elif self.time_steps <= 6:
                reward -= 1
        return reward

    def get_state(self):
        # the state is the number of units I have
        return self.units_gained

    def is_terminated(self):
        return self.terminated

    def get_all_time_steps(self):
        return self.all_time_steps

    def get_all_state_counts(self):
        return len(list(range(0, self.maxUnits+1)))

    def get_all_action_count(self):
        return len(list(range(1, self.maxUnits-1)))

    def get_all_sap_count(self):
        return self.get_all_action_count() * self.get_all_state_counts()

    def get_all_states(self):
        return list(range(0, self.maxUnits+1))

    def plot_state_best_action(self):
        # plots the states with the Wager
        plt.plot([row[0] for row in self.state_best], [row[1] for row in self.state_best])
        plt.show()



