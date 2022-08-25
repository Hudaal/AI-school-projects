import numpy as np
import math
import random


class MCST:
    """ The monte carlo search tree class  """
    def __init__(self, root_state):
        self.root_node = Node(root_state, action=None, parent=None)

    def tree_policy_bestUCT(self, node, player_sign, game):
        # This function will control the tree policy
        best = -float('inf') * player_sign
        chosen_action = None
        for action, child in node.children.items():
            if child.visited_count == 0: # not visited yet
                uct = child.k * float('inf')
            else:
                uct = (child.Q / child.visited_count) + \
                      (child.k * math.sqrt(2 * math.log(child.parent.visited_count) / child.visited_count))

            # get the action based on the player, min for player 2 and max for player 1
            if player_sign >= 0:
                if uct > best:
                    best = uct
                    chosen_action = action
            else:
                if uct < best:
                    best = uct
                    chosen_action = action
        if chosen_action is None:
            if len(list(self.root_node.children.keys())) > 0:
                chosen_action = random.choice(list(self.root_node.children.keys()))
            else:
                if len(game.get_legal_actions()) > 0:
                    chosen_action = random.choice(game.get_legal_actions())
        return chosen_action

    def tree_search(self, game):
        # Tree search from root node to leaf one
        visited_node = self.root_node
        while len(list(visited_node.children.keys())) > 0:
            action = self.tree_policy_bestUCT(visited_node, game.get_player_sign(), game)
            if action is None:
                return visited_node
            if action in visited_node.children.keys():
                new_state = game.do_action(action)
                visited_node.children[action].state = new_state
                visited_node = visited_node.children[action]
        return visited_node

    def node_expantion(self, game, visited_node):
        # expand the tree with this node
        if (visited_node.visited_count != 0 or visited_node.parent is None) and not game.is_final_state():
            for action in game.get_legal_actions():
                new_state = game.get_unapplied_state(action)
                new_Node = Node(new_state, action=action, parent=visited_node)
                visited_node.add_child(action, new_Node)
                visited_node.is_not_leaf = True
            # get first expanded child and continue with it
            expanded_nodes = list(visited_node.children.values())
            expanded_node = expanded_nodes[0]
            expanded_node.is_not_leaf = False
            return expanded_node
        else: return visited_node

    def rollout(self, state, actor, game):
        while not game.is_final_state():
            legal_actions_to_be_sent = game.get_legal_actions_on_all()
            # get the action based on default policy from actor
            action = actor.default_policy(state, legal_actions_to_be_sent)
            state = game.do_action(action)

    def backpropagation(self, visited_node, game):
        # update the visited count in each node and the reward got from the game
        while visited_node is not None: # is not the root node
            visited_node.visited_count += 1
            visited_node.Q += game.get_reward()
            visited_node = visited_node.parent

    def distribution(self, all_actions_count):
        # Returns a list of normalized distribution for all children of the root node
        distribution = []
        for action in range(1, all_actions_count+1):
            if action in self.root_node.children.keys():
                distribution.append(self.root_node.children[action].visited_count / self.root_node.visited_count)
            else:
                distribution.append(0.0)
        distribution = np.array(distribution)
        normalized = [float(d) / sum(distribution) for d in distribution]
        return tuple(normalized)

    def set_new_root(self, action, new_state):
        # set new root to the tree
        if action in self.root_node.children.keys():
            self.root_node.children[action].state = new_state
            self.root_node = self.root_node.children[action]
            self.root_node.parent = None

    def tree_steps(self, actor, simulated_game):
        # Run the tree algorithm 1) search, 2) expand, 3) rollout, 4) backpropagation
        leaf_node = self.tree_search(simulated_game)
        expanded_node = self.node_expantion(simulated_game, leaf_node)
        if expanded_node == leaf_node:
            # print('the node is not expanded')
            pass
        leaf_node = expanded_node
        self.rollout(leaf_node.state, actor, simulated_game)
        self.backpropagation(leaf_node, simulated_game)


class Node:
    """ The tree node """
    def __init__(self, state, action=None, parent=None, const=1):
        self.state = state
        self.action = action
        self.children = {}
        self.parent = parent
        self.visited_count = 0
        self.Q = 0
        self.k = const
        self.is_not_leaf = False

    def add_child(self, action, child_node):
        self.children[action] = child_node


