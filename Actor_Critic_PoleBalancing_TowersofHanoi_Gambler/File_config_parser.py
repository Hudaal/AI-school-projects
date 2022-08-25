class File_config_parser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.global_dict = {}
        self.cartpole_dict = {}
        self.hanoi_dict = {}
        self.gambler_dict = {}

        self.valid_global_keys = ['episodes','max_T', 'critic_t', 'critic_n', 'layers', 'lr_c_n', 'lr_c_t', 'lr_a',
                                  'decay_a', 'decay_c', 'discount_a', 'discount_c', 'epsilon', 'decay_e', 'display']
        self.valid_cartpole_keys = ['L', 'mp', 'gravity', 'timestep', 'play']
        self.valid_hanoi_keys = ['pegs', 'discs', 'play']
        self.valid_gambler_keys = ['pw', 'play']
        self.layers_count = 0

    def parse_config_file(self):
        file = open(self.file_name, mode='r')
        glob = False
        cart = False
        hanoi = False
        gambler = False
        layer_dict = {}
        for line in file:
            if 'GLOBALS' in line:
                glob = True
                cart = False
                hanoi = False
                gambler = False
            if 'CartPole' in line:
                glob = True
                cart = True
                hanoi = False
                gambler = False
            if 'Hanoi' in line:
                glob = False
                cart = False
                hanoi = True
                gambler = False
            if 'Gambler' in line:
                glob = False
                cart = False
                hanoi = False
                gambler = True
            else:
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                splited_line = line.split(',')
                for item in splited_line:
                    key_value_list = item.split(':')
                    if len(key_value_list) > 1:
                        if '(' in key_value_list[1] and ')' in key_value_list[1]:
                            key_value_list[1] = key_value_list[1].replace('(', '')
                            key_value_list[1] = key_value_list[1].replace(')', '')
                            values = key_value_list[1].split(';')
                            values = [int(values[i]) for i in range(len(values)) if values[i]]
                        else:
                            values = key_value_list[1]
                        if glob and key_value_list[0] in self.valid_global_keys:
                            self.global_dict[key_value_list[0]] = values
                        elif cart and key_value_list[0] in self.valid_cartpole_keys:
                            self.cartpole_dict[key_value_list[0]] = values
                        elif hanoi and key_value_list[0] in self.valid_hanoi_keys:
                            self.hanoi_dict[key_value_list[0]] = values
                        elif gambler and key_value_list[0] in self.valid_gambler_keys:
                            self.gambler_dict[key_value_list[0]] = values

if __name__ == '__main__':
    F = File_config_parser('config.txt')
    F.parse_config_file()
    print(F.cartpole_dict)
    print(F.global_dict)
    print(F.hanoi_dict)
    print(F.gambler_dict)