from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    """Abstract environment class used as a common interface for different environments/simworlds.

    Interface for environments:
    * State is a tuple.
        The interface should be a tuple even if the internals of the environment deals with one dimension.
    * Actions is an integer.
        This should account for all possible actions, and doesn't care if actions are illegal, but the environment
        must implement functionality to check which actions are legal in given states. Applications using the
        environment could check which actions are legal, but the environments can also punish if illegal actions
        are taken. Illegal actions should not move the state of the environment.
    """

    def __init__(self, n_timesteps: int = 2000, store_states: bool = False):
        """
        :param n_timesteps: Max number of timesteps that can be performed before environment is terminated.
        """
        self.store_states: bool = store_states
        self.n_timesteps: int = n_timesteps

        self.state_action_size = sum(self.state_shape) + self.actions

    @abstractmethod
    def initialize(self) -> tuple:
        """Initializes environment/state and returns the initialized state.

        :return: The initial state.
        """

        raise NotImplementedError('Subclasses must implement initialize()')

    @abstractmethod
    def next(self, action: int) -> tuple[tuple, float, bool]:
        """Applies action to the environment, moving it to the next state.

        :param action: The action to perform
        :return: (next_state, reward, finished)
                    next_state: the current state of the environment after applying the action
                    reward: a numerical reward for moving to the state
                    finished: boolean specifying if the environment has reached some terminal condition
        """

        raise NotImplementedError('Subclasses must implement next()')

    @abstractmethod
    def action_legal_in_state(self, action: int, state: tuple):
        """Checks whether an action is legal in a given state.

        :param action: Action to check.
        :param state: State to check.
        :return: Whether the action is legal in the given state.
        """

        raise NotImplementedError('Subclasses must implement action_legal_in_state()')

    def get_legal_actions(self, state: tuple) -> np.ndarray:
        return np.array([action for action in range(self.actions) if self.action_legal_in_state(action, state)])

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """The shape of the state space.

        :return: A tuple describing the shape of the state space.
        """

        raise NotImplementedError('Subclasses must implement state_shape property')

    @property
    @abstractmethod
    def actions(self) -> int:
        """The actions that can be performed.

        :return: Number of total actions.
        """

        raise NotImplementedError('Subclasses must implement actions property')

    @abstractmethod
    def visualize(self, vis_sleep: float = 1.0) -> None:
        """Visualizes the state history.

        :param vis_sleep: Seconds between each frame (if needed).
        """

        raise NotImplementedError('Subclasses must implement visualize() class method')


    def encode_state_action(self, state: tuple, action: int) -> np.ndarray:
        """Encodes a tupled state and action int to a bit list.

        Example: (2,3,4), 3 -> [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1)] (Parenthesis for visualizing each integer)

        :param state: State in the form of a tuple.
        :param action: Action in the form of an integer.
        :return: State/action in the form of a bit list.
        """

        onehot_state_action = []
        for s, l in zip(state, self.state_shape):
            onehot_state = [0]*l
            onehot_state[s] = 1
            onehot_state_action.extend(onehot_state)
        onehot_action = [0]*self.actions
        onehot_action[action] = 1
        onehot_state_action.extend(onehot_action)
        return np.array(onehot_state_action, dtype=int)


