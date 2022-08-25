import numpy as np
import copy


class Hex:
    """ The HEX game"""
    def __init__(self, board_dim=4):
        self.board_dim = board_dim
        self.board_size = board_dim ** 2
        self.init_board()

    def init_board(self):
        """ Make an empty game board """
        # There is two states in the game, 1) is 2D matrix with the player number or 0, 2) Network of nodes to check
        # neighbors
        self.board = np.zeros((self.board_dim, self.board_dim))
        self.Nodes = {}
        self.player = 1
        # This is the start and end edges of each player [start player 1, end player 1, start player 2, end player 2]
        self.last_nodes_coord = [[(0, i) for i in range(self.board_dim)],
                                 [(self.board_dim - 1, i) for i in range(self.board_dim)],
                                 [(i, 0) for i in range(self.board_dim)],
                                 [(i, self.board_dim - 1) for i in range(self.board_dim)]]
        for row in range(self.board_dim):
            for col in range(self.board_dim):
                is_last = False
                for list_nodes in self.last_nodes_coord:
                    if (row, col) in list_nodes:
                        is_last = True
                        break
                # Create the empty nodes with al neighbors
                self.Nodes[row, col] = (HexNode((row, col), self, is_last=is_last))
        for key in self.Nodes.keys():
            self.Nodes[key].add_neighbors()
        self.terminated = False
        self.reward = 0
        self.winner = 0
        self.done_actions = []
        # The action are the position of chosen place in the board with row and col
        self.all_actions = self.get_all_actions()

    def get_game_dim(self):
        # This returns the input and output size of the ANET
        return self.board_dim ** 2, self.board_dim ** 2

    def is_final_state(self):
        return self.terminated

    def get_player(self):
        return self.player

    def get_player_sign(self):
        # player 1 gives positive sign, and player 2 gives negative
        if self.player == 1:
            return 1
        else:
            return -1

    def get_init_state(self):
        return np.array(self.board).flatten()

    def get_state(self):
        return np.array(self.board).flatten()

    def fix_if_flatten(self, state):
        # If the state is flatten, make is as 2d matrix with rows and columns
        if len(state) == self.board_dim:
            return np.array(state)
        else:
            count = 0
            board = np.zeros((self.board_dim, self.board_dim))
            for i in range(self.board_dim):
                for j in range(self.board_dim):
                    board[i][j] = state[count]
                    count += 1
            return board

    def get_all_actions(self):
        # returns a list of all actions before start playing
        all_actions = []
        for row in range(self.board_dim):
            for col in range(self.board_dim):
                all_actions.append((row, col))
        return all_actions

    def get_legal_actions(self, state=None):
        # returns a list of the indexes of the possible actions
        legal_actions = []
        # l = []
        if state is None:
            state = self.board
        for row in range(self.board_dim):
            for col in range(self.board_dim):
                if state[row][col] == 0:
                    # l.append((row, col))
                    legal_actions.append(self.all_actions.index((row, col)) + 1)
        # print('legal actions', l)
        return legal_actions

    def get_legal_actions_on_all(self, state=None):
        # returns a list with length of all actions and non possible actions as zero and possible actions are the
        # indexes of the actions
        if state is None:
            state = self.board
        state = self.fix_if_flatten(state)
        to_send_actions = []
        all_actions = list(range(len(self.all_actions)))
        legal_actions = self.get_legal_actions(state)
        for action in all_actions:
            if action + 1 in legal_actions:
                to_send_actions.append(action + 1)
            else:
                to_send_actions.append(0)
        return to_send_actions

    def do_action(self, action_index):
        # apply the action with the given index
        row, col = self.all_actions[action_index - 1]
        if self.board[row][col] != 0:
            return np.array(self.board).flatten()
        # update the value of the given row and column
        self.board[row][col] = self.player
        self.Nodes[(row, col)].value = self.player
        for row_col in self.Nodes.keys():
            self.Nodes[row_col].add_to_neighbors(self.Nodes[(row, col)], self.player)
        self.done_actions.append((row, col))
        coord_idx = 0 if self.player == 1 else 2 # set the right start and end edges for this player
        for r, c in self.last_nodes_coord[coord_idx]:
            # check the positions in the edges and traverse their neighbors to check if the current player won
            self.terminated = self.traverse_(self.Nodes[r, c], coord_idx)
            if self.terminated:
                self.winner = self.player
                if self.winner == 1:
                    self.reward += 1
                else:
                    self.reward -= 1
                return np.array(self.board).flatten()
        self.player = (self.player % 2) + 1 # update the player
        return np.array(self.board).flatten()

    def traverse_(self, Node, start=0, visited=None):
        # traverse the board from start edg to end one
        terminated = False
        if visited is None:
            visited = []
        if Node in visited:
            return False
        visited.append(Node)
        if self.board[Node.row][Node.col] == self.player:
            if (Node.row, Node.col) in self.last_nodes_coord[start + 1]:
                return True
            for neighbor in Node.neighbors.keys():
                if terminated:
                    return True
                terminated = self.traverse_(Node.neighbors[neighbor], start, visited)
            return terminated
        return False

    def get_winner_id(self):
        return self.winner

    def get_reward(self):
        return self.reward

    def get_unapplied_state(self, action_index):
        # Get next state if we apply this action, but now apply the action
        row, col = self.all_actions[action_index - 1]
        copied_board = copy.deepcopy(self.board)
        copied_board[row][col] = self.player
        return np.array(copied_board).flatten()

    def undo_move(self):
        row, col = self.done_actions.pop()
        self.board[row][col] = 0
        self.Nodes[(row, col)].value = 0
        # remove neighbors
        self.player = (self.player % 2) + 1
        return np.array(self.board).flatten()

    def show_board(self, board=None):
        # visualize the board
        print('2: \\    1: /\n')
        if board is None:
            board = self.board
        board = self.fix_if_flatten(board)
        lines = [[] for line_num in range(self.board_dim*2 - 1)]
        for line_num in range(self.board_dim*2 - 1):
            for row, col in self.Nodes.keys():
                if row+col == line_num:
                    lines[line_num].append((row, col))
                    lines[line_num].sort()
                    lines[line_num] = list(reversed(lines[line_num]))

        for i, line in enumerate(lines):
            print(" " * (((self.board_dim - len(line)) * 2) + 1), end=' ')
            for row, col in line:
                if (row, col) != lines[i][-1]:
                    end = '--'
                else:
                    end = ' '
                print(board[row][col], end=end)
            print()
            print(" " * ((self.board_dim - len(line)) * 2), end=' ')
            for row, col in line:
                if line != lines[-1]:
                    if i < self.board_dim - 1:
                        print(" / \\", end=' ')
                    elif i >= self.board_dim-1:
                        if (row, col) == lines[i][-1]:
                            print(" / ", end=' ')
                        elif (row, col) == lines[i][0]:
                            print("   \\", end=' ')
                        else:
                            print(" / \\", end=' ')
            print()



class HexNode:
    """ The Hex node class with some variables and neighbors """
    def __init__(self, x_y, board, is_last=False, value=0, connected=None):
        if connected is None:
            connected = [False, False]
        self.board = board
        self.value = value
        self.row, self.col = x_y
        self.neighbors = {}
        self.is_last = is_last
        self.connected = connected
        self.last_for_player = [0, 0]
        # The positions of my neighbors
        self.row_col_neighbors = [(self.row - 1, self.col), (self.row + 1, self.col), (self.row, self.col - 1),
                                  (self.row, self.col + 1), (self.row - 1, self.col + 1), (self.row + 1, self.col - 1)]

    def add_neighbors(self):
        # Add all neighbors to this node as empty node
        for row, col in self.row_col_neighbors:
            if 0 <= row < self.board.board_dim and 0 <= col < self.board.board_dim:
                self.neighbors[row, col] = self.board.Nodes[row, col]

    def add_to_neighbors(self, Node, player, stop=False):
        # add this node to the neighbors list if it's position is in my positions list
        if (Node.row, Node.col) in self.row_col_neighbors:
            self.neighbors[(Node.row, Node.col)] = Node
            self.connected[player - 1] = True
