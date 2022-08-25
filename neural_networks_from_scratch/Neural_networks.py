import numpy as np
from Layer import Layer
from Data_generator import Data_generator
import matplotlib.pyplot as plt

from image_viewer import Viewer


class NeuralNetwork:

    def __init__(self, input_dim, hidden_layer, neurons, lr, epochs, activation_functions, output_type, loss_type,
                 reg_type, minibatch_size, reg_rate, weight_intervals, bias_intervals, print_) -> None:
        self.neurons = neurons
        self.lr = lr
        self.epochs = epochs
        self.train_data = None
        self.test_data = None
        self.valid_data = None
        self.train_targets = None
        self.test_targets = None
        self.valid_targets = None
        self.input_dim = input_dim
        self.output_type = output_type
        self.loss_type = loss_type
        self.reg_type = reg_type
        self.reg_rate = reg_rate
        self.minibatch_size = minibatch_size
        self.print_ = print_
        self.layers_count = 2 + hidden_layer  # Number of layers of the NN
        self.layers = []
        self.d_activation_functions = []
        self.activation_functions = []
        self.softmax_act = []

        for f in activation_functions:
            if f == 'sigmoid':
                self.activation_functions.append(self.sigmoid)
                self.d_activation_functions.append(self.d_sigmoid_f)
            elif f == 'tanh':
                self.activation_functions.append(self.tanh)
                self.d_activation_functions.append(self.d_tanh)
            elif f == 'linear':
                self.activation_functions.append(self.linear)
                self.d_activation_functions.append(self.d_linear)
            elif f == 'relu':
                self.activation_functions.append(self.relu)
                self.d_activation_functions.append(self.d_relu)
            elif f == 'softmax':
                self.activation_functions.append(self.soft_max)
                self.d_activation_functions.append(None)
            else:
                # raise TypeError
                print('TypeError')
        for l in range(self.layers_count):
            if l == 0:
                self.layers.append(Layer(l, 0, neurons[l], self.activation_functions[l],
                                         self.d_activation_functions[l], lr[l]))
            else:
                self.layers.append(Layer(l, neurons[l-1], neurons[l], self.activation_functions[l],
                                     self.d_activation_functions[l], lr[l], weights_low=weight_intervals[l][0],
                                         weights_hight=weight_intervals[l][1],
                                         bias_init=bias_intervals[0][l]))

    def load_data(self, DG) -> None:
        # Load from image generator, DG is the data generator
        self.train_data, self.train_targets = DG.fletting_into_vectors(DG.train_shapes, self.minibatch_size)
        self.test_data, self.test_targets = DG.fletting_into_vectors(DG.test_shapes, self.minibatch_size)
        self.valid_data, self.valid_targets = DG.fletting_into_vectors(DG.validate_shapes, self.minibatch_size)

    def Reg_L1(self):
        # Regulator L1
        sum_all = 0
        for layer in self.layers:
            sum_all += np.sum(np.abs(layer.weights)) # + np.sum(np.abs(layer.bias))
        return sum_all * self.reg_rate

    def Reg_L2(self):
        # Regulator L2
        sum_all = 0
        for layer in self.layers:
            sum_all += np.sum(np.square(layer.weights)) # + np.sum(layer.bias ** 2)
        return sum_all * self.reg_rate

    def mean_squared_error(self, y, pred):
        # The loss function MSE
        if self.reg_type == 'L1':
            return np.mean(np.square(np.subtract(y, pred))) #+ self.Reg_L1() * self.reg_rate
        elif self.reg_type == 'L2':
            return np.square(np.subtract(y, pred)).mean() #+ self.Reg_L2() * self.reg_rate
        return False

    def d_mse(self, y, pred, n):
        # The derivation of MSE
        return 2 / n * (y - pred)

    def d_cross_entropy(self, y, pred):
        # The derivation of cross entropy
        all_col = []
        for i, col in enumerate(pred):
            if col == 0:
                all_col.append(- col / y[i])
            else:
                all_col.append(y[i]/col)
        return np.array(all_col)

    def cross_entropy(self, y, pred):
        # Cross entropy loss function
        return (-1 * np.sum(y * np.log(pred + 1e-9)))/len(y)

    def sigmoid(self, x):
        # activation function g(x)
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        # The derivative of g(x) => g'(x) = g(x) (1 - g(x))
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def d_sigmoid_f(self, f):
        return np.exp(f) / (np.exp(f) + 1)**2

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, f):
        return 1 / (np.cosh(f) + 1) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, f):
        return f > 0

    def linear(self, x):
        return x

    def d_linear(self, f):
        return np.ones(len(f))

    def soft_max(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def validate(self):
        """This function compare the validation targets with the predicted values from the validation data set"""
        predictions = self.forward_pass(self.valid_data)
        equal_sum = 0
        targets = []
        for index, pred in enumerate(predictions):
            pred = [round(pred[i]) for i in range(len(pred))]
            targets.append([round(self.valid_targets[index][i]) for i in range(len(pred))])
            if pred == targets[index]:
                equal_sum += 1
        print('equal_sum, in the validation set with the target', equal_sum, 'from', len(self.valid_targets))

    def train(self) -> None:
        """Train the network with forward and backward pass for every minibatch"""
        for i in range(0, self.layers_count):
            self.layers[i].init_weights()
            self.layers[i].init_bias()
            self.layers[i].init_activations()
        loss_l = []
        loss_valid_l = []
        for e in range(self.epochs):  # how many times to repeat
            for i, x in enumerate(self.train_data):
                pred = self.forward_pass(x)
                self.backward_pass(i)

            if self.loss_type == 'mse':
                if self.output_type == 'softmax':
                    loss = self.mean_squared_error(self.train_targets, self.predict(self.train_data)) * 1000 # It's multiplied by 1000 to show the changes clearer in the plot
                    loss_valid = self.mean_squared_error(self.valid_targets, self.predict(self.valid_data)) * 1000 + 1
                else:
                    loss = self.mean_squared_error(self.train_targets, self.predict(self.train_data))
                    loss_valid = self.mean_squared_error(self.valid_targets, self.predict(self.valid_data))
            else:
                loss = self.cross_entropy(self.train_targets, self.predict(self.train_data))
                loss_valid = self.cross_entropy(self.valid_targets, self.predict(self.valid_data))
            # This part should add the regulator to the loss function, but it gave wrong answers and
            # I hadn't enough time to fix it
            '''if self.reg_type == 'L2': reg_loss = self.Reg_L2()
            else: reg_loss = self.Reg_L1()
            loss += reg_loss
            loss_valid += loss'''

            # append the loss for the training and validation set to plot them and print them
            loss_l.append(loss)
            loss_valid_l.append(loss_valid)
            if e % 10 == 0:
                if self.print_:
                    print("Loss: " + str(loss), "in epoch = ", e)
                    print("Loss: " + str(loss_valid), "in epoch = ", e)
                    self.validate()
                    print('input: ', self.train_data[i])
                    print('output predicted: ', pred)
                    print('target: ', self.train_targets[i])

        # The process viewer
        fig, ax = plt.subplots()
        plt.title('NN')
        ax.plot(range(self.epochs), loss_l, label='Training data')
        ax.plot(range(self.epochs), loss_valid_l, label='Validating data')
        leg = ax.legend()
        plt.show()

    def backward_pass(self, index):
        predicted_output = self.layers[-1].activations
        if self.output_type == 'softmax':
            predicted_output = self.softmax_act # This matrix have the results when the softmax function has been called in forward pass
        if self.loss_type == 'mse':
            J_loss_out = self.d_mse(self.train_targets[index], predicted_output, self.layers[-1].output_neurons_count)
        else: # cross entropy
            J_loss_out = self.train_targets[index] - self.layers[-1].activations

        if self.output_type == 'softmax' and self.loss_type == 'mse':
            # softmax back propagation
            J_soft_out = np.zeros((self.layers[-1].output_neurons_count, self.layers[-1].output_neurons_count))
            for i in range(len(J_soft_out)):
                for j in range(len(J_soft_out)):
                    if i != j:
                        J_soft_out[i][j] = -predicted_output[i] * predicted_output[j]
                    else:
                        J_soft_out[i][i] = predicted_output[i] - predicted_output[i] ** 2
            J_loss_out = np.dot(J_loss_out, J_soft_out)

        for layer_reversed in reversed(range(self.layers_count - 1)):
            J_loss_out = self.layers[layer_reversed + 1].backward_pass(self.layers[layer_reversed].activations,
                                                                       J_loss_out, self.reg_rate)

    def forward_pass(self, x):
        self.layers[0].activations = x
        for i in range(1, self.layers_count):
            self.layers[i].activations = self.layers[i].forward_pass(self.layers[i-1].activations)
        if self.output_type == 'softmax': # The output layer
            self.softmax_act = self.soft_max(self.layers[-1].activations)
            # return self.soft_max(self.layers[-1].activations)
        return self.layers[-1].activations

    def predict(self, x):
        return self.forward_pass(x)

    def test(self):
        pred = self.predict(self.test_data)
        if self.loss_type == 'mse':
            if self.output_type == 'softmax':
                loss = self.mean_squared_error(self.test_targets, pred) * 1000
            else:
                loss = self.mean_squared_error(self.test_targets, pred)
        else:
            loss = self.cross_entropy(self.test_targets, pred)
        return loss, self.test_targets, pred


if __name__ == '__main__':
    nn_class = NeuralNetwork
    n_features = 30
    rows_count = 20
    column_count = 20
    minibatch_size = 1
    viewer = Viewer(ndim=20, width=20, height=20, noise=3,
                    image_count_in_set=500)
    viewer.generateImages()
    #viewer.viewImages()
    DG = viewer.get_image_sets()
    DG.generate_split_image_sets(70, 20, 10)
    network = nn_class(input_dim=n_features, hidden_layer=2, neurons=[rows_count*column_count*minibatch_size, 100, 50, 4],
                       lr=[0.02, 0.02, 0.02, 0.02], epochs=100,
                       activation_functions=['sigmoid', 'relu', 'relu', 'sigmoid']
                       , output_type='none', loss_type='mse', reg_type='L2', minibatch_size=minibatch_size,
                       reg_rate=0.05, weight_intervals=[[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]], bias_intervals=[[0, 0, 0, 0]], print_=True)

    network.load_data(DG)
    network.train()
    loss, targets, pred = network.test()
    print('The loss result from the testing set is ', loss)
    print('The targets are: ')
    for i, p in enumerate(pred):
        print('Target: ', targets[i], 'and the predicted output: ', p)


