from typing import Optional

import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from learner.critics.critic import Critic
from learner.lite_model import LiteModel


class NetworkCritic(Critic):
    """Critic using a keras neural network."""

    def __init__(self,
                 layer_sizes: list[int],
                 loss_function: tfk.losses.Loss = tfk.losses.mse,
                 learning_rate: float = 0.001,
                 *args, **kwargs):
        """
        :param layer_sizes: Hidden layer sizes. Each entry in the list specifies the size of a hidden layer.
        :param batch_size: How many losses should be accumulated before stepping the optimizer.
        """

        super().__init__(*args, **kwargs)
        self.model = tfk.Sequential([
            tfkl.Input(shape=(self.environment.state_action_size, )),
            *[tfkl.Dense(units=size, activation='relu') for size in layer_sizes],
            tfkl.Dense(units=1, activation='linear')
        ])

        self.model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), loss=loss_function)
        self.model.summary()

        self.replay_buffer = []

    def Q(self, state: tuple, action: int, lite_model: LiteModel = None) -> float:
        network_input = self.environment.encode_state_action(state, action)
        if lite_model is None:
            return self.model.predict(np.expand_dims(network_input, axis=0))
        else:
            return lite_model.predict_single(network_input)

    def update_Q(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.train_on_batch(x, y)
