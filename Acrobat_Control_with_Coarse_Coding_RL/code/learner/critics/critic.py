from abc import ABC, abstractmethod

import numpy as np

from environments.environment import Environment


class Critic(ABC):
    """Abstract critic class which should be inherited from."""

    def __init__(self,
                 environment: Environment,
                 discount: float = 0.7):
        """
        :param environment: Environment object that the critic observes.
        :param discount: Discount parameter used to train V(S).
        """

        self.environment: Environment = environment
        self.discount: float = discount

    @abstractmethod
    def update_Q(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError('Subclasses must implement update_Q()')