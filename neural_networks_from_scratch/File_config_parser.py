class File_config_parser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.global_dict = {}
        self.gen_dict = {}
        self.layers_dict_list = []
        self.valid_global_keys = ['loss', 'lrate', 'wreg', 'wrt', 'minibatchsize', 'epochs', 'verbos']
        self.valid_layer_keys = ['input', 'type', 'act', 'wr', 'lrate', 'br', 'size']
        self.valid_generator_keys = ['count', 'height', 'width', 'ndim', 'noise', 'test', 'valid', 'train']
        self.layers_count = 0

    def parse_config_file(self):
        file = open(self.file_name, mode='r')
        glob = False
        layers = False
        gen = False
        layer_dict = {}
        for line in file:
            if 'GLOBALS' in line:
                glob = True
                layers = False
                gen = False
            if 'LAYERS' in line:
                glob = False
                layers = True
                gen = False
            if 'GENERATOR' in line:
                gen = True
                glob = False
                layers = False
            else:
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                splited_line = line.split(',')
                if layers:
                    self.layers_count += 1
                for item in splited_line:
                    key_value_list = item.split(':')
                    if len(key_value_list) > 1:
                        if '(' in key_value_list[1] and ')' in key_value_list[1]:
                            key_value_list[1] = key_value_list[1].replace('(', '')
                            key_value_list[1] = key_value_list[1].replace(')', '')
                            values = key_value_list[1].split(';')
                            values = [float(values[i]) for i in range(len(values)) if values[i]]
                        else:
                            values = key_value_list[1]
                        if glob and key_value_list[0] in self.valid_global_keys:
                            self.global_dict[key_value_list[0]] = values
                        elif layers and key_value_list[0] in self.valid_layer_keys:
                            layer_dict[key_value_list[0]] = values
                        elif gen and key_value_list[0] in self.valid_generator_keys:
                            self.gen_dict[key_value_list[0]] = values
                if layer_dict:
                    self.layers_dict_list.append(layer_dict)
                    layer_dict = {}


if __name__ == '__main__':
    F = File_config_parser('config.txt')
    F.parse_config_file()
    print(F.layers_dict_list)
    print(F.global_dict)
    print(F.gen_dict)