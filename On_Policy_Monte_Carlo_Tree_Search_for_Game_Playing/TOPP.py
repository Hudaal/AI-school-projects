import numpy as np
from ANET import ANET
from Hex import Hex
import parameters


class TOPP:
    """  The Tournament of Progressive Policies """
    def __init__(self, games_count, game, agent_count):
        self.games_count = games_count
        self.game = game
        self.agent_count = agent_count
        self.agents_score = [0 for _ in range(agent_count)]
        self.agents_lose = [0 for _ in range(agent_count)]
        self.agents = []
        input_nn_size, output_nn_size = game.get_game_dim()
        for i in range(agent_count):
            self.agents.append(ANET(input_nn_size, output_nn_size, load_saved=True))
            self.agents[i].load_saved_model(i)

        self.c = 0
        self.serie_count = 0

    def run_all_games(self):
        all_winners = []
        # start from player 1 -> before last one
        for first in range(self.agent_count - 1):
            agent1 = self.agents[first]
            for second in range(first + 1, self.agent_count):
                # from the next agent to last one
                agent2 = self.agents[second]
                ply_num = 0
                self.serie_count += 1
                for game_idx in range(self.games_count):
                    self.c += 1
                    winner = self.play_game(agent1, agent2)
                    if parameters.show_board:
                        self.game.show_board()
                        print('winner is: ', winner)
                    # If we have swaped the players, w will hold the right number based on first and second agent
                    # to update the winning and loosing lists
                    w = winner if ply_num % 2 == 0 else (winner % 2) + 1
                    all_winners.append(winner)
                    if w == 1:
                        self.agents_score[first] += 1
                        self.agents_lose[second] += 1
                    else:
                        self.agents_score[second] += 1
                        self.agents_lose[first] += 1
                    ply_num += 1
                    # swap the players so every player has the possibility to start
                    temp = agent1
                    agent1 = agent2
                    agent2 = temp
                all_winners = []
                print('The winning score after serie', self.serie_count, self.agents_score)
        print('scores',  self.agents_score)
        print('lose', self.agents_lose)
        print(self.c)

    def play_game(self, player1, player2):
        self.game.init_board()
        state = self.game.get_init_state()
        player_num = 0
        while not self.game.is_final_state():
            to_send_actions = self.game.get_legal_actions_on_all()
            player = player1 if player_num % 2 == 0 else player2
            action = player.default_policy(state, to_send_actions, get_best=True)
            state = self.game.do_action(action)
            player_num += 1
        return self.game.get_winner_id()


def main():
    topp = TOPP(parameters.game_count, Hex(parameters.TOPP_dim), parameters.agents_M)
    topp.run_all_games()

main()