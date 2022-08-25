"""
Contains replay buffer class used during RL with MCT.
"""

from typing import Optional
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Utility class used to maintain a replay buffer.
    """

    def __init__(self, targets: list[str], batch_size: int = 32, buffer_size: Optional[int] = None):
        """
        :param batch_size: Batch size retrieved from the replay buffer to be used to train the actor
        :param buffer_size: Max buffer size. If None, the buffer size is unlimited
        """

        self._state_actions: deque[np.ndarray] = deque([], maxlen=buffer_size)
        self._targets: dict = {target: deque([], maxlen=buffer_size) for target in targets}
        self._n: int = 0
        self.batch_size: int = batch_size
        self.buffer_size: int = buffer_size

    def store_replay(self, state_actions: np.ndarray, **targets: any) -> None:
        """
        Stores replay (state, targets pairs) in internal buffer
        :param states: List of states to be associated with targets
        :param targets: Arbitrary number of targets can be added as a replay. All targets much match length of states
        """

        self._state_actions.append(state_actions)
        for k, v in targets.items():
            self._targets[k].append(v)

        self._n += 1

    def store_replays(self, state_actions: list[np.ndarray], **targets: list[any]) -> None:
        """
        Stores replays (state, targets pairs) in internal buffer
        :param states: List of states to be associated with targets
        :param targets: Arbitrary number of targets can be added as a replay. All targets much match length of states
        """

        self._state_actions.extend(state_actions)
        for k, v in targets.items():
            self._targets[k].extend(v)

        self._n += len(state_actions)

    @property
    def n(self):
        """
        Returns current size of replay buffer.
        :return: Current size of replay buffer
        """

        return min(self._n, len(self._state_actions))

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the replay buffer is ready.
        The replay buffer is ready if there are more (x,y) pairs in the replay buffer than the provided batch size.
        :return: Whether the replay buffer is ready
        """

        return self.n >= self.batch_size

    def get_batch(self, target: str, replace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Retieves a random batch from the replay buffer.
        :param target: Which target to retrieve batch for
        :param replace: Whether an item can be retrieved multiple times
        :return: A random batch from the replay buffer
        """

        if target not in self._targets:
            raise KeyError(f'Target {target} has not been initialized')

        idxs = np.random.choice(self.n, size=self.batch_size, replace=replace)
        return np.array(self._state_actions)[idxs], np.array(self._targets[target])[idxs]

    def get_sample(self, target: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Retieves a random sample from the replay buffer.
        :return: A random sample from the replay buffer
        """

        if target not in self._targets:
            raise KeyError(f'Target {target} has not been initialized')

        idx = np.random.randint(self.n)
        return np.array(self._state_actions[idx]), np.array(self._targets[target][idx])