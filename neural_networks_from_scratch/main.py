from Data_generator import Data_generator
from image_viewer import Viewer
from Neural_networks import NeuralNetwork
from File_config_parser import File_config_parser

class main_class:
    def __init__(self):
        name = input('The config file name: ')
        file_parser = File_config_parser(name)
        file_parser.parse_config_file()
        self.neurons = []
        self.activation_functions = []
        self.lr = []
        self.wr = []
        self.br = []
        self.input_dim = 0
        self.output_type = 'none'
        self.layer_count = len(file_parser.layers_dict_list)
        self.hidden_layers = self.layer_count - 2
        self.DG = None
        self.network = None
        self.global_vars(file_parser.global_dict)
        self.layers_vars(file_parser.layers_dict_list)
        self.generator_vars(file_parser.gen_dict)

    def global_vars(self, global_dict):
        if 'lrate' in global_dict.keys():
            self.global_lr = float(global_dict['lrate'])
        else:
            self.global_lr = 0.1
        if 'loss' in global_dict.keys():
            self.loss_type = global_dict['loss']
        else:
            self.loss_type = 'mse'
        if 'wreg' in global_dict.keys():
            self.reg_rate = float(global_dict['wreg'])
        else:
            self.reg_rate = 'mse'
        if 'minibatchsize' in global_dict.keys():
            self.minibatchsize = int(global_dict['minibatchsize'])
        else:
            self.minibatchsize = 1
        if 'wrt' in global_dict.keys():
            self.reg_type = global_dict['wrt']
        else:
            self.reg_type = 'L2'
        if 'epochs' in global_dict.keys():
            self.epochs = int(global_dict['epochs'])
        else:
            self.epochs = 200
        if 'verbos' in global_dict.keys():
            self.verbos = False if global_dict['verbos'] == '0' else True
        else:
            self.verbos = True

    def layers_vars(self, layers_dict_list):
        c = 0
        for layer in layers_dict_list:
            if c == 0 and 'input' in layer.keys():
                self.input_dim = int(layer['input'])
                self.neurons.append(self.input_dim)
                self.activation_functions.append('linear')
                self.lr.append(self.global_lr)
                self.wr.append([-0.1, 0.1])  # First element will not be used
                self.br.append(0)  # First element will not be used
            else:
                if 'size' in layer.keys():
                    self.neurons.append(int(layer['size']))
                else:
                    self.neurons.append(1)
                if 'act' in layer.keys():
                    self.activation_functions.append(layer['act'])
                else:
                    self.activation_functions.append('sigmoid')
                if 'wr' in layer.keys():
                    self.wr.append(layer['wr'])
                else:
                    self.wr.append([-0.1, 0.1])
                if 'lrate' in layer.keys():
                    self.lr.append(float(layer['lrate']))
                else:
                    self.lr.append(self.global_lr)
                if 'br' in layer.keys():
                    self.br.append(int(layer['br']))
                else:
                    self.br.append(0)
                if c == self.layer_count - 1 and 'type' in layer.keys():
                    self.output_type = layer['type']
                    self.activation_functions[c] = 'sigmoid'
                    self.neurons[c] = self.minibatchsize * 4
                    self.br.append(0)
            c += 1

    def generator_vars(self, gen_dict):
        if 'count' in gen_dict.keys():
            self.image_set_count = int(gen_dict['count'])
        else:
            self.image_set_count = 100
        if 'height' in gen_dict.keys():
            self.image_height = int(gen_dict['height'])
        else:
            self.image_height = 20
        if 'width' in gen_dict.keys():
            self.image_width = int(gen_dict['width'])
        else:
            self.image_width = 20
        if 'ndim' in gen_dict.keys():
            self.ndim = int(gen_dict['ndim'])
        else:
            self.ndim = 20
        if 'noise' in gen_dict.keys():
            self.noise = int(gen_dict['noise'])
        else:
            self.noise = 5
        if 'train' in gen_dict.keys():
            self.train_persent = int(gen_dict['train'])
        else:
            self.train_persent = 70
        if 'test' in gen_dict.keys():
            self.test_persent = int(gen_dict['test'])
        else:
            self.test_persent = 10
        if 'valid' in gen_dict.keys():
            self.valid_persent = int(gen_dict['valid'])
        else:
            self.valid_persent = 20

    def generate_view_images(self, view=True):
        viewer = Viewer(ndim=self.ndim, width=self.image_width, height=self.image_height, noise=self.noise,
                        image_count_in_set=self.image_set_count)
        viewer.generateImages()
        if view: viewer.viewImages()
        self.DG = viewer.get_image_sets()

    def load_data_to_NN(self):
        self.DG.generate_split_image_sets(self.train_persent, self.valid_persent, self.test_persent)
        input_ = self.image_width * self.image_height * self.minibatchsize
        self.neurons[0] = input_
        self.network = NeuralNetwork(input_dim=input_, hidden_layer=self.hidden_layers, neurons=self.neurons,
                       lr=self.lr, epochs=self.epochs,
                       activation_functions=self.activation_functions
                       , output_type=self.output_type, loss_type=self.loss_type, reg_type=self.reg_type, minibatch_size=self.minibatchsize,
                       reg_rate=self.reg_rate, weight_intervals=self.wr, bias_intervals=[self.br], print_=self.verbos)

        self.network.load_data(self.DG)
        self.network.train()
        loss, targets, pred = self.network.test()
        print('The loss result from the testing set is ', loss)
        if self.verbos:
            print('The targets are: ')
            for i, p in enumerate(pred):
                print('Target: ', targets[i], 'and the predicted output: ', p)

if __name__ == '__main__':
    main_class_ = main_class()
    main_class_.generate_view_images()
    main_class_.load_data_to_NN()
    # main_class_.network.train()