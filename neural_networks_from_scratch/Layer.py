import numpy as np

class Layer:
    def __init__(self, layer_number, input_neurons_count, output_neurons_count,
                 activation_function, d_activation_function, lr, weights_low=-0.1, weights_hight=0.1, bias_init=0):
        self.weights_hight = weights_hight
        self.weights_low = weights_low
        self.bias = bias_init

        self.layer_number = layer_number
        self.input_neurons_count = input_neurons_count
        self.output_neurons_count = output_neurons_count
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.lr = lr
        self.weights = []
        self.activations = []
        self.bias = []

    def init_weights(self):
        if self.layer_number == 0:
            return
        self.weights = np.random.uniform(low=self.weights_low, high=self.weights_hight,
                                         size=(self.input_neurons_count, self.output_neurons_count))

    def init_activations(self):
        self.activations = np.zeros(self.output_neurons_count)

    def init_bias(self):
        self.bias = 0

    def forward_pass(self, prev_activations):
        layer_output = np.dot(prev_activations, self.weights) + self.bias
        return self.activation_function(layer_output)

    def backward_pass(self, prev_activations, J_loss_out, reg_rate):
        self.weights = np.array(self.weights)

        # Fill the diagonal with the derivation of the activation function
        J_sum_out = np.zeros((self.output_neurons_count, self.output_neurons_count))
        np.fill_diagonal(J_sum_out, [self.d_activation_function(self.activations)])

        J_out_weights = np.outer(prev_activations, J_sum_out.diagonal())

        J_loss_weights = np.dot(J_out_weights, J_loss_out)
        J_out_prev = np.dot(J_sum_out, self.weights.transpose())
        # self.weights += self.lr * J_loss_weights.reshape(J_loss_weights.shape[0], -1)
        d = J_loss_out * J_sum_out.diagonal()

        self.weights += self.lr * (d.reshape(d.shape[0], -1) * prev_activations).transpose() # * reg_rate
        self.bias += self.lr * J_loss_out * J_sum_out.diagonal() # * reg_rate
        J_loss_out = np.dot(J_loss_out, J_out_prev)
        return J_loss_out.reshape(np.array(prev_activations).shape)