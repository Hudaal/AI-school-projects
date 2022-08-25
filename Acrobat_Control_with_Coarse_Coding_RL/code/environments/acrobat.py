import math
import csv
import os

import numpy as np

from environments.environment import Environment


class Acrobat(Environment):

    def __init__(self,
                 bins,
                 m_1: float = 1.0,
                 m_2: float = 1.0,
                 L_1: float = 1.0,
                 L_2: float = 1.0,
                 L_c1: float = 0.5,
                 L_c2: float = 0.5,
                 g: float = 9.81,
                 timestep: float = 0.05,
                 no_tiles: int = 3,
                 n_collected_timesteps: int = 2,
                 initial_eps: float = 0.9,
                 eps_decay: float = 0.97,
                 output_dir: str = 'decent',
                 *args,
                 **kwargs):
        """
        :param n_timesteps: Max number of timesteps that can be performed before environment is terminated.
        """
        self.bins = bins
        self.m_1 = m_1
        self.m_2 = m_2
        self.L_1 = L_1
        self.L_2 = L_2
        self.L_c1 = L_c1
        self.L_c2 = L_c2
        self.g = g
        self.timestep = timestep
        self.state = None
        self.no_tiles = no_tiles
        self.n_collected_timesteps = n_collected_timesteps
        super().__init__(*args, **kwargs)
        self.xp1 = 0
        self.yp1 = 0
        self.xp2 = 0
        self.yp2 = -self.L_1
        self.tilings = []
        self.inside_state = [0, 0, 0, 0]
        self.timesteps_count = 0
        self.create_tilings(
            [[-math.pi, math.pi], [-5, 5], [-math.pi, math.pi], [-5, 5]]
            , self.bins, [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1], [1.5, 1.5, 1.5, 1.5]])

        self.state_history = []

    def create_tiling(self, feat_range, bins, offset, i, max_offset):
        """Creates 1 tiling spec of 1 dimension(feature)

        feat_range: feature range; example: [-1, 1]
        bins: number of bins for that feature; example: 10
        offset: offset for that feature; example: 0.2
        """

        tile = np.linspace(feat_range[0]-max_offset/bins, feat_range[1], bins + 1) + offset/bins
        tile = tile[1:-1]

        return tile

    def create_tilings(self, feature_ranges, bins, offsets):
        """Create tilings or coarse coding.

        :param feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        :param bins: bin size for each tiling
        :param offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """
        # for each tiling
        for tile_i in range(self.no_tiles):
            tiling_bin = bins[tile_i]
            tiling_offset = offsets[tile_i]

            tiling = []
            # for each feature dimension
            for feat_i in range(len(feature_ranges)):
                max_offset = np.array(offsets)[:, feat_i].max()
                feat_range = feature_ranges[feat_i]
                # tiling for 1 feature
                feat_tiling = self.create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i], feat_i, max_offset)
                tiling.append(feat_tiling)
            self.tilings.append(tiling)
        return np.array(self.tilings)

    def get_tile_encoding(self, state):
        """Transforms continuous state into coarse coding representation

        :param state: a state that needs to be encoded
        :param return: the encoding for the feature on each layer
        """
        num_dims = len(state)
        state_codings = []
        for tiling in self.tilings:
            for i in range(num_dims):
                feature = state[i]
                tiling_i = tiling[i]  # tiling on that dimension
                coding_i = np.digitize(feature, tiling_i)  # note: returns the indices of the bins
                state_codings.append(coding_i)
        return tuple(state_codings)


    def initialize(self) -> tuple:
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """
        theta_1 = 0
        d_theta_1 = 0
        theta_2 = 0
        d_theta_2 = 0
        self.xp1 = 0
        self.yp1 = 0
        self.xp2 = 0
        self.yp2 = -self.L_1
        self.inside_state = [theta_1, d_theta_1, theta_2, d_theta_2]
        self.state = self.get_tile_encoding(self.inside_state)
        self.timesteps_count = 0
        self.state_history = []

        if self.store_states:
            self.state_history.append(self.inside_state)
        return self.state

    def bind_theta(self, theta):
        if theta > math.pi:
            return theta - 2 * math.pi
        elif theta < -math.pi:
            return theta + 2 * math.pi
        else:
            return theta


    def next(self, action: int) -> tuple[tuple, float, bool]:
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """
        reward = -1
        finished = False
        force = [-1, 0, 1]
        for timestep in range(self.n_collected_timesteps):
            theta_1 = self.inside_state[0]
            theta_2 = self.inside_state[2]
            d_theta_1 = self.inside_state[1]
            d_theta_2 = self.inside_state[3]

            # intermediaries
            phi_2 = self.m_2 * self.L_c2 * self.g * math.cos(theta_1 + theta_2 - math.pi/2)
            phi_1 = -self.m_2 * self.L_1 * self.L_c2 * d_theta_2**2 * math.sin(theta_2) \
                    - 2 * self.m_2 * self.L_1 * self.L_c2 * d_theta_1 * d_theta_2 * math.sin(theta_2) \
                    + (self.m_1 * self.L_c1 + self.m_2 * self.L_1) * self.g * math.cos(theta_1 - math.pi/2) \
                    + phi_2

            d_2 = self.m_2 * (self.L_c2**2 + self.L_1 * self.L_c2 * math.cos(theta_2)) + 1
            d_1 = self.m_1 * self.L_c1**2 \
                  + self.m_2 * (self.L_1**2 + self.L_c2**2 + 2 * self.L_1 * self.L_c2 * math.cos(theta_2)) \
                  + 2

            dd_theta_2 = (force[action] + phi_1 * d_2/d_1 - self.m_2 * self.L_1 * self.L_c2 * d_theta_1**2 * math.sin(theta_2) - phi_2) \
                         / (self.m_2 * self.L_c2**2 + 1 - (d_2**2)/d_1)
            dd_theta_1 = - (d_2 * dd_theta_2 + phi_1)/d_1

            d_theta_2 = d_theta_2 + self.timestep * dd_theta_2
            d_theta_1 = d_theta_1 + self.timestep * dd_theta_1
            theta_2 = self.bind_theta(theta_2 + self.timestep * d_theta_2)
            theta_1 = self.bind_theta(theta_1 + self.timestep * d_theta_1)

            self.inside_state = [theta_1, d_theta_1, theta_2, d_theta_2]
            self.timesteps_count += 1
            if self.store_states:
                self.state_history.append(self.inside_state)
            self.state = self.get_tile_encoding(self.inside_state)
            _, ytip = self.find_location_coordinates()
            if ytip >= self.L_2:
                finished = True
                reward += 10
                print('Won')
                break
            if self.timesteps_count >= self.n_timesteps:
                finished = True
                print('Lost')
                break

        return self.state, reward, finished

    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """
        if action in [0, 1, 2]:
            return True  # er vel alltid lov
        else:
            return False

    def find_location_coordinates(self) -> tuple:
        theta3 = self.inside_state[0] + self.inside_state[2]
        self.xp2 = self.xp1 + self.L_1 * math.sin(self.inside_state[0])
        self.yp2 = self.yp1 - self.L_1 * math.cos(self.inside_state[0])
        xtip = self.xp2 + self.L_2 * math.sin(theta3)
        ytip = self.yp2 - self.L_2 * math.cos(theta3)
        return xtip, ytip

    @property
    def state_shape(self) -> tuple:
        """The shape of the state space

        :return: A tuple describing the shape of the state space.
        """
        state_shape = []
        for i in range(4):
            for j in range(self.no_tiles):
                state_shape.append(self.bins[i][j])
        return tuple(state_shape)

    @property
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions.
        """
        return 3

    def dump_thetas(self, path: str) -> None:
        """Dump thetas to csv file.

        :param path: Path to csv file
        """
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['theta1', 'theta2', 'dtheta1', 'dthetat2'])
            for state in self.state_history:
                writer.writerow([state[0], state[2], state[1], state[3]])


    def visualize(self, vis_sleep: float = 1.0) -> None:
        """Visualizes the state history.

        :param vis_sleep: Seconds between each frame (if needed).
        """

        self.dump_thetas('test.csv')

