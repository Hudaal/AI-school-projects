# Hex game
dim = 4
actor = None
game = None
board_size = dim ** 2
number_actual_game = 100
number_search_games = 15
save_model_idx = 10
batch_size = 50

# ANET
learning_rate = 0.001
epsilon = 0.1
epsilon_decay = 0.01
hidden_layers_count = 2
hidden_dims = [40, 10]
activation = 'relu' # linear, sigmoid, tanh, relu
optimizer = 'Adagrad' # SGD, Adagrad, RMSprop, Adam

# TOPP
agents_M = 5
game_count = 25
TOPP_dim = 4

show_board = True
