import os
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from environments.environment import Environment
from learner.critics.network_critic import NetworkCritic
from learner.lite_model import LiteModel
from learner.replay_buffer import ReplayBuffer

timestamp = datetime.now().strftime('%y%m%d-%H%M%S')

class ReinforcementLearner:
    """Class implementing the SARSA algorithm with a critic and an environment."""

    def __init__(self,
                 environment: Environment,
                 critic: NetworkCritic,
                 output_dir: str = os.path.join('output', timestamp)):
        """
        :param environment: The Environment object the actor/critic operates on
        :param critic: A Critic object
        """

        self.environment: Environment = environment
        self.critic: NetworkCritic = critic
        self.critic_lite: LiteModel = LiteModel.from_keras_model(critic.model)

        self.state_lengths = tuple([len(format(s, 'b')) for s in self.environment.state_shape])
        self.action_length = len(format(self.environment.actions, 'b'))

        self.steps: Optional[np.ndarray] = None

        self.output_dir = output_dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)


    def get_action_distribution(self, state: tuple) -> np.ndarray:
        """
        :param environment: The Environment object the actor/critic operates on
        :param critic: A Critic object
        """
        distribution = np.zeros(self.environment.actions)
        for action in range(self.environment.actions):
            if self.environment.action_legal_in_state(action, state):
                value = self.critic.Q(state, action, self.critic_lite)
                distribution[action] = value
        exp = np.exp(distribution)
        return exp / exp.sum()

    def epsilon_greedy_action(self, state: tuple, epsilon: float) -> int:
        if np.random.rand() < epsilon + 0.05:
            return np.random.choice(self.environment.get_legal_actions(state))
        else:
            return int(np.argmax(self.get_action_distribution(state)))

    def fit(self, n_episodes: int = 300, initial_eps: float = 1, eps_decay: float = 0.97, batch_size: int=512, buffer_size: int=512) -> None:
        """Fits the tables/networks of the actors and critic by learning from the environment

        :param n_episodes: Number of episodes to run the environment.
        :return:
        """

        self.steps = np.zeros(n_episodes, dtype=int)
        replay_buffer = ReplayBuffer(targets=['Q'], batch_size=batch_size, buffer_size=buffer_size)
        for episode in range(n_episodes):
            self.critic_lite = LiteModel.from_keras_model(self.critic.model)
            #epsilon = initial_eps * (1 - episode/n_episodes)
            epsilon = initial_eps * eps_decay**episode
            print(epsilon)
            state = self.environment.initialize()
            action = self.epsilon_greedy_action(state, epsilon)

            finished = False
            while not finished:
                self.steps[episode] += 1
                next_state, reward, finished = self.environment.next(action)
                next_action = self.epsilon_greedy_action(next_state, epsilon)

                target = reward + self.critic.discount * self.critic.Q(next_state, next_action, self.critic_lite)
                replay_buffer.store_replay(self.environment.encode_state_action(state, action), Q=target)

                state = next_state
                action = next_action

            if replay_buffer.is_ready:
                x, y = replay_buffer.get_batch('Q')
                self.critic.update_Q(x, y)

            print(f'Finished episode {episode} after {self.steps[episode]} steps')

            if self.environment.store_states and finished:
                self.environment.dump_thetas(os.path.join(self.output_dir, f'episode{episode}.csv'))

    def visualize_fit(self) -> None:
        """Visualizes the number of steps taken at each episode during the last fit."""

        plt.plot(self.steps)
        plt.title(f'Timesteps taken in {self.environment.__class__.__name__}')
        plt.xlabel('Episode')
        plt.ylabel('Timesteps')
        plt.show()

    def run(self, visualize: bool = True, vis_sleep: float = 1.0):
        """Runs through the environment using the fitted actor/critic.

        :param visualize: Whether or not the state history of the environment should be visualized.
        :return:
        """

        steps: int = 0
        state = self.environment.initialize()
        self.environment.store_states = visualize
        finished = False
        while not finished:
            steps += 1
            action = self.epsilon_greedy_action(state, 0)
            state, _, finished = self.environment.next(action)

        print(f'Finished run after {steps} steps')
        if visualize:
            self.environment.visualize(vis_sleep)
