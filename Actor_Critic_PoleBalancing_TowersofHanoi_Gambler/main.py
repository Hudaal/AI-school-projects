from File_config_parser import File_config_parser
from CartPol_SimWorld import cartPol_SimWorld
from TowersofHanoiSimWorld import TowersofHanoi_SimWorld
from Gambler_SimWorld import Gambler_SimWorld
from Agent import Agent
from Actor import Actor
from Critic_table_based import Critic_table_based
from Critic_NN_based import Critic_NN_based

class main_class:
    def __init__(self):
        name = input('The config file name: ')
        file_parser = File_config_parser(name)
        file_parser.parse_config_file()
        self.nn_layers = []
        self.episodes = 0
        self.max_T = 0
        self.critic_t = 0
        self.critic_n = 0
        self.lr_c_n = 0
        self.lr_c_t = 0
        self.lr_a = 0
        self.decay_a = 0
        self.decay_c = 0
        self.discount_a = 0
        self.discount_c = 0
        self.epsilon = 0
        self.decay_e = 0
        self.display = 0
        self.global_vars(file_parser.global_dict)
        self.L = 1
        self.mp = 0
        self.gravity = 0
        self.time_step = 0
        self.play_c = 0
        self.cart_vars(file_parser.cartpole_dict)
        self.play_h = 0
        self.pegs = 0
        self.discs = 2
        self.hanoi_vars(file_parser.hanoi_dict)
        self.play_g = 0
        self.pw = 0
        self.gambler_vars(file_parser.gambler_dict)


    def global_vars(self, global_dict):
        if 'episodes' in global_dict.keys():
            self.episodes = int(global_dict['episodes'])
        else:
            self.episodes = 200
        if 'max_T' in global_dict.keys():
            self.max_T = int(global_dict['max_T'])
        else:
            self.max_T = 300
        if 'critic_t' in global_dict.keys():
            self.critic_t = True if int(global_dict['critic_t']) != 0 else False
        else:
            self.critic_t = True
        if 'critic_n' in global_dict.keys():
            self.critic_n = True if int(global_dict['critic_n']) != 0 else False
        else:
            self.critic_n = False
        if 'layers' in global_dict.keys():
            self.nn_layers = global_dict['layers']
        else:
            self.nn_layers = [4, 2]
        if 'lr_c_n' in global_dict.keys():
            self.lr_c_n = float(global_dict['lr_c_n'])
        else:
            self.lr_c_n = 0.01
        if 'lr_c_t' in global_dict.keys():
            self.lr_c_t = float(global_dict['lr_c_t'])
        else:
            self.lr_c_t = 0.001
        if 'lr_a' in global_dict.keys():
            self.lr_a = float(global_dict['lr_a'])
        else:
            self.lr_a = 0.09
        if 'decay_a' in global_dict.keys():
            self.decay_a = float(global_dict['decay_a'])
        else:
            self.decay_a = 0.7
        if 'decay_c' in global_dict.keys():
            self.decay_c = float(global_dict['decay_c'])
        else:
            self.decay_c = 0.7
        if 'decay_e' in global_dict.keys():
            self.decay_e = float(global_dict['decay_e'])
        else:
            self.decay_e = 0.0001
        if 'discount_c' in global_dict.keys():
            self.discount_c = float(global_dict['discount_c'])
        else:
            self.discount_c = 0.7
        if 'discount_a' in global_dict.keys():
            self.discount_a = float(global_dict['discount_a'])
        else:
            self.discount_a = 0.7
        if 'epsilon' in global_dict.keys():
            self.epsilon = float(global_dict['epsilon'])
        else:
            self.epsilon = 0.01
        if 'display' in global_dict.keys():
            self.display = True if int(global_dict['display']) != 0 else False
        else:
            self.display = True


    def cart_vars(self, gen_dict):
        if 'L' in gen_dict.keys():
            self.L = float(gen_dict['L'])
        else:
            self.L = 0.5
        if 'mp' in gen_dict.keys():
            self.mp = float(gen_dict['mp'])
        else:
            self.mp = 0.1
        if 'gravity' in gen_dict.keys():
            self.gravity = float(gen_dict['gravity'])
        else:
            self.gravity = -9.8
        if 'time_step' in gen_dict.keys():
            self.time_step = float(gen_dict['time_step'])
        else:
            self.time_step = 0.02
        if 'play' in gen_dict.keys():
            self.play_c = True if int(gen_dict['play']) != 0 else False
        else:
            self.play_c = False

    def hanoi_vars(self, gen_dict):
        if 'pegs' in gen_dict.keys():
            self.pegs = int(gen_dict['pegs'])
        else:
            self.pegs = 3
        if 'discs' in gen_dict.keys():
            self.discs = int(gen_dict['discs'])
        else:
            self.discs = 3
        if 'play' in gen_dict.keys():
            self.play_h = True if int(gen_dict['play']) != 0 else False
        else:
            self.play_h = False

    def gambler_vars(self, gen_dict):
        if 'pw' in gen_dict.keys():
            self.pw = float(gen_dict['pw'])
        else:
            self.pw = 0.4
        if 'play' in gen_dict.keys():
            self.play_g = True if int(gen_dict['play']) != 0 else False
        else:
            self.play_g = False

    def play_games(self):
        actor = Actor(self.lr_a, self.discount_a, self.decay_a, self.epsilon, self.decay_e)
        if self.play_c:
            if self.critic_t:
                critic = Critic_table_based(self.lr_c_t, self.discount_c, self.decay_c)
            else:
                critic = Critic_NN_based(self.lr_c_n, self.discount_c, self.decay_c, [1, 1], 1,
                                         len(self.nn_layers), self.nn_layers)
            env = cartPol_SimWorld(self.L, self.mp, self.gravity, self.time_step, self.max_T, self.display)
            agent = Agent(env, actor, critic, self.critic_t)
            agent.train(self.episodes, self.display)

            env.plot_best_timestep_angle()
            env.plot_timesteps_angles(self.episodes)

        elif self.play_h:
            env = TowersofHanoi_SimWorld(self.pegs, self.discs, self.max_T, self.display)
            if self.critic_t:
                critic = Critic_table_based(self.lr_c_t, self.discount_c, self.decay_c)
            else:
                critic = Critic_NN_based(self.lr_c_n, self.discount_c, self.decay_c, [self.discs, 2], 1,
                                         len(self.nn_layers), self.nn_layers)
            agent = Agent(env, actor, critic, self.critic_t)
            agent.train(self.episodes, self.display)

            env.plot_timesteps_episodes(self.episodes)
            env.plot_best_game()

        else: # The Gambler
            env = Gambler_SimWorld(self.pw, self.max_T, self.display)
            if self.critic_t:
                critic = Critic_table_based(self.lr_c_t, self.discount_c, self.decay_c)
            else:
                critic = Critic_NN_based(self.lr_c_n, self.discount_c, self.decay_c, [1, 1], 1,
                                         len(self.nn_layers), self.nn_layers)
            agent = Agent(env, actor, critic, self.critic_t)
            agent.train(self.episodes, self.display)

            env.plot_state_best_action()


if __name__ == '__main__':
    main_class_ = main_class()
    main_class_.play_games()
