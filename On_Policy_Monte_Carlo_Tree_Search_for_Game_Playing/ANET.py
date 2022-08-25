import numpy as np
import random
import tensorflow as tf
import parameters


class ANET:
    """ The actor NN """
    def __init__(self, input_dim, output_dim, load_saved=False, load_num=1):
        self.epsilon = parameters.epsilon
        self.epsilon_decay = parameters.epsilon_decay
        self.lr = parameters.learning_rate
        self.hidden_layers_count = parameters.hidden_layers_count
        self.hidden_dims = parameters.hidden_dims
        self.save_counter = 0
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create the model with the given input and output sizes
        self.input_layer = tf.keras.Input(shape=input_dim)
        self.next_layer = tf.keras.layers.Flatten()(self.input_layer)
        for layer in range(self.hidden_layers_count):
            self.next_layer = tf.keras.layers.Dense(self.hidden_dims[layer], activation=parameters.activation)(self.next_layer)

        self.output_layer = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.softmax)(self.next_layer)
        self.model = tf.keras.Model(self.input_layer, self.output_layer, name='ANET')
        self.model.compile(optimizer=parameters.optimizer, loss='mse')
        self.model.summary()

        # if load_saved:
        #    self.load_saved_model(load_num)

    def save_model(self):
        self.model.save('models/{}.h5'.format(self.save_counter))
        self.save_counter += 1

    def load_saved_model(self, num):
        self.model = tf.keras.models.load_model("models/{}/{}.h5".format(int(np.sqrt(self.input_dim)), num), compile=False)

    def normalize(self, list_):
        if sum(list_) == 0:
            return list_
        list_sum = sum(list_)
        return list_ / list_sum

    def fit(self, minibatch):
        states = []
        probabilities = []
        # The minibatch is list of lists of states and targets
        for state, D in minibatch:
            states.append(np.array(state))
            probabilities.append(np.array(D))
        self.model.fit(np.array(states), np.array(probabilities), batch_size=20) # Fit the model

    def predict_best(self, inputs, legal_moves):
        # predict the list of actions from the given state as input, and multiply with the legal moves
        # list witch contains zeros if the action is not possible
        pred = self.model.predict(np.array([inputs]), verbose=False)
        summed_pred = sum(pred)
        action_prob = summed_pred * np.array(legal_moves)
        action_prob /= np.sum(action_prob)
        best_action = np.argmax(action_prob) # get the highest value action
        return (best_action + 1) % self.output_dim

    def predict_action(self, inputs, legal_moves):
        pred = self.model.predict(np.array([inputs]))
        summed_pred = sum(pred)
        best_action = np.argmax(summed_pred)
        return (best_action + 1) % self.output_dim

    def get_distribution(self, inputs, moves):
        pred = self.model.predict(np.array(inputs))
        summed_pred = sum(pred)
        action_prob = summed_pred * np.array(moves)
        action_prob /= np.sum(action_prob)
        return action_prob

    def default_policy(self, state, legal_moves, get_best=False):
        # if get_best is True, the algorithm will predict the best move without the greedy check
        if random.random() < self.epsilon and not get_best:
            if len(legal_moves) > 0:
                return random.choice([move for move in legal_moves if move > 0])
            else:
                return False
        moves = [] # 1 or 0
        for move in legal_moves:
            if move == 0:
                moves.append(move)
            else:
                moves.append(1)
        return self.predict_best(state, moves)
