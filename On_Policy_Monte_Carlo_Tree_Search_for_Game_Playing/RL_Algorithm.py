import numpy as np
import random
import copy
from ANET import ANET
from MCST import MCST
import parameters


class RLAlgorithm:
    """ The game playing """
    def __init__(self, game, state_size, number_actions, number_actual_game, number_search_games,
                 save_model_idx, batch_size):
        self.actual_game = game
        self.replay_buffer = []
        self.list_of_buffers = []
        self.ANET = ANET(state_size, number_actions)
        self.buffer_insertion_index = 0
        self.number_actual_game = number_actual_game
        self.number_search_games = number_search_games
        self.save_model_idx = save_model_idx
        self.all_actions_count = len(self.actual_game.get_all_actions())
        self.batch_size = batch_size

    def run_all_actual_games(self):
        # Runs the RL algorithm
        for game_a in range(self.number_actual_game):
            self.actual_game.init_board()
            root_state = self.actual_game.get_init_state()
            mcst = MCST(root_state)
            while not self.actual_game.is_final_state():
                for search_game_idx in range(self.number_search_games):
                    simulated_game = copy.deepcopy(self.actual_game) # get a copy of the game with the current root node
                    mcst.tree_steps(self.ANET, simulated_game)
                    search_game_idx += 1
                    if search_game_idx == self.number_search_games:
                        break
                # Actual game
                D = mcst.distribution(self.all_actions_count)
                self.replay_buffer.append((tuple(root_state), D))
                self.buffer_insertion_index += 1
                to_send_actions = self.actual_game.get_legal_actions_on_all()
                action = self.ANET.default_policy(self.actual_game.get_state(), to_send_actions)
                new_state = self.actual_game.do_action(action)

                if parameters.show_board:
                    self.actual_game.show_board()

                # mcst.set_new_root(action, new_state)
                mcst = MCST(new_state)
                root_state = new_state

            print('Winning player', self.actual_game.winner)
            random_minibatch_RBUF = self.get_random_minibatch_replay_buffer()
            print('................ FIT ....................')
            self.ANET.fit(random_minibatch_RBUF)
            if game_a % self.save_model_idx == 0:
                self.ANET.save_model()

    def get_random_minibatch_replay_buffer(self):
        return random.sample(self.replay_buffer, min(self.buffer_insertion_index, self.batch_size))
        # return random.choice(self.list_of_buffers)
