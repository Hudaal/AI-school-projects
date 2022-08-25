from operator import ne
import tensorflow as tf
from splitgd import SplitGD
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

class Critic_NN_based:
    """The Neuron based critic"""
    def __init__(self, lr, disount_f, decay_f, state_size, action_size, layer_count, neurons):
        self.lr = lr
        self.discount_f = disount_f
        self.decay_f = decay_f
        self.e = {}
        self.v = {}
        self.state_size = state_size
        self.action_size = action_size
        self.layer_count = layer_count
        self.neurons = neurons # a list with all layers and number of neurons in each layer

        self.model = Sequential()
        for i in range(layer_count): # adding the layers to the model
            self.model.add(Dense(neurons[i], activation=tf.nn.relu))
        self.model.add(Dense(1, activation=tf.nn.relu))
        self.model.build(input_shape=(self.state_size[0], self.state_size[1]))
        self.model.summary()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))

    def fit_nn(self, state, target, epochs=1, verbose=0):
        # This will be called after each step to fit the network
        self.model.fit(np.array(state).reshape(self.state_size[0], self.state_size[1]), np.array(target), epochs, verbose)

    def get_state_value(self, state):
        # Get the predicted value from the network
        values = self.model.predict(np.array(state).reshape(self.state_size[0], self.state_size[1]))
        return values