from turtle import position
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

class cartPol_SimWorld:
    def __init__(self, L, mp, gravity, time_step, max_T=300, display=False) -> None:
        self.T = max_T
        self.gravity = gravity
        self.pole_mass = mp
        self.pole_length = L
        self.timestep = time_step
        self.display = display
        self.all_time_steps = []
        self.round_grade_ex = 10000
        self.round_grade = 4
        self.all_angels = []
        self.reset()

    def reset(self):
        self.cart_mass = 1
        self.angle_derivation = 0
        self.angle_2derivation = 0
        self.position = 0
        self.velocity = 0
        self.acc = 0
        self.F = 10
        self.B = 10
        self.angle_limits = [-0.21, 0.21]
        self.angle = np.random.uniform(self.angle_limits[0], self.angle_limits[1])
        self.movment_position_bounds = [-2.4, 2.4]
        self.T = 300
        self.action = 1
        self.time_step_count = 0
        self.terminated = False
        self.lost = 0
        self.angles = []

    def get_init_state(self):
        # random angle in the limits
        self.angle = np.random.uniform(self.angle_limits[0], self.angle_limits[1])
        self.angles.append(self.angle)
        self.position = 0 # center position
        self.angle_derivation = 0

        angle = round(self.angle, self.round_grade)
        pos = round(self.position, 1)
        return angle if angle != 0 else 0

    def get_all_actions(self):
        return [-1, 1] # -F or +F

    def get_all_possible_actions(self):
        return self.get_all_actions() # all actions are possible

    def do_action(self, action_sign):
        self.B = action_sign * self.F
        reward = 0
        # updated the values with the chosen action
        one_part = self.pole_mass * self.pole_length * self.angle_derivation**2 * sin(self.angle)
        two_mass = self.pole_mass+self.cart_mass
        angle_2derivation_1 = self.gravity * sin(self.angle) + ((cos(self.angle) * (-self.B - one_part)) / two_mass)
        angle_2derivation_2 = self.pole_length * (4/3 - ((self.pole_mass*(cos(self.angle)**2))/two_mass))
        self.angle_2derivation = angle_2derivation_1 / angle_2derivation_2
        self.acc = (self.B + self.pole_mass*self.pole_length*(self.angle_derivation**2 * sin(self.angle) - self.angle_2derivation*cos(self.angle))) / two_mass
        self.angle_derivation += self.angle_2derivation * self.timestep
        self.velocity += self.timestep * self.acc
        self.angle += self.timestep * self.angle_derivation
        self.position += self.timestep * self.velocity
        if self.display:
            print('step: ', self.time_step_count, 'position: ', self.position, 'angle: ', self.angle)
        reward += 0.5 # one more step gives +0.5 reward

        self.angles.append(self.angle)

        if -1.4 <= self.position <= 1.4:
            reward += 10 # almost in the center
        elif -2.0 <= self.position <= 2.0:
            reward += 4 # near the limits, lower reward
        elif -2.4 <= self.position <= 2.4:
            reward -= 4 # near the limits, lower reward
        else:
            reward -= 13 # out the limit, - reward
            self.lost = 1
        if -0.1 <= self.angle <= 0.1:
            reward += 10 # almost in the center
        elif -0.16 <= self.angle <= 0.16:
            reward += 4 # near the limits, lower reward
        elif -0.21 <= self.angle <= 0.21:
            reward = -4 # near the limits, lower reward
        else:
            reward -= 13 # out the limit, - reward
            self.lost = 1

        self.time_step_count += 1
        if self.lost:
            self.terminated = True
            self.all_time_steps.append(self.time_step_count)
            self.all_angels.append(self.angles)
            return reward

        if self.time_step_count == self.T:
            reward += 10 # all time steps visited
            self.terminated = True
            self.all_time_steps.append(self.time_step_count)
            self.time_step_count = 0
            self.all_angels.append(self.angles)
            return reward

        return reward


    def get_state(self):
        # I round the angle to get lower possible states
        # the angle is the state
        angle = round(self.angle, self.round_grade)
        pos = round(self.position, 1)
        return angle if angle != 0 else 0

    def is_terminated(self):
        return self.terminated

    def get_all_time_steps(self):
        return self.all_time_steps

    def get_all_state_counts(self):
        return len(range(int(self.angle_limits[0]*self.round_grade_ex), int(self.angle_limits[1]*self.round_grade_ex) + 1))
               # * len(range(int(self.movment_position_bounds[0] * 10), int(self.movment_position_bounds[1] * 10) + 1))

    def get_all_action_count(self):
        return len(self.get_all_actions())

    def get_all_sap_count(self):
        return self.get_all_action_count() * self.get_all_state_counts()

    def get_all_states(self):
        all_states = []
        for an in range(int(self.angle_limits[0]*self.round_grade_ex), int(self.angle_limits[1]*self.round_grade_ex) + 1):
            #for pos in range(int(self.movment_position_bounds[0] * 10), int(self.movment_position_bounds[1] * 10) + 1):
            all_states.append((an/self.round_grade_ex))
        return all_states

    def get_angle(self):
        return self.angle

    def plot_timesteps_angles(self, episods):
        # plot the time steps with the angle achieved in each one
        fig, ax = plt.subplots()
        plt.plot(range(episods), self.get_all_time_steps())
        ax.set_xlabel('episodes')
        ax.set_ylabel('Time steps')
        plt.show()

    def plot_best_timestep_angle(self):
        # plot the angle changes in the best game
        max_timestep = max(self.all_time_steps)
        max_index = self.all_time_steps.index(max_timestep)
        best_angels = self.all_angels[max_index]
        fig, ax = plt.subplots()
        plt.plot(range(max_timestep), best_angels[:-1])
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Angel')
        plt.show()

