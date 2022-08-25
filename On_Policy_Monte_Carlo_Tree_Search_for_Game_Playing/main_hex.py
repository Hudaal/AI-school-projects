from MCST import MCST
from ANET import ANET
from Hex import Hex
from RL_Algorithm import RLAlgorithm
import parameters

if __name__ == '__main__':
    game = Hex(parameters.dim)
    learner = RLAlgorithm(game, parameters.board_size, len(game.get_legal_actions()), parameters.number_actual_game,
                          parameters.number_search_games, parameters.save_model_idx, parameters.batch_size)
    learner.run_all_actual_games()