import numpy as np
import matplotlib.pyplot as plt

class TowersofHanoi_SimWorld:
    def __init__(self, pegs_count=4, discs_count=3, max_T=300, display=False) -> None:
        self.all_time_steps = []
        self.pegs_count = pegs_count
        self.discs_count = discs_count
        self.max_T = max_T
        self.display = display
        self.discs = [i for i in range(self.discs_count)]
        self.all_games = []
        self.reset()
        self.act = []
        self.one_game = ()
        self.c = []

    def reset(self):
        self.time_step_count = 0
        self.terminated = False
        self.lost = 0
        self.state = []
        self.pegs = [[i, []] for i in range(self.pegs_count)] # the array inside each peg will have the discs order
        self.init_pig = 0
        self.top_discs = []
        self.done_actions = []
        self.done_actions_count = 0
        self.ac = []

    def get_init_state(self):
        # random peg with all discs on it with the right order
        self.init_peg = np.random.randint(0, self.pegs_count)
        for order in reversed(range(len(self.discs))):
            # the state is list of discs count with small lists [peg the disc on, order of the disc]
            self.state.append((self.init_peg, order))
            self.pegs[self.init_peg][1].append(order)
        self.top_discs.append(0)
        self.one_game += (tuple([tuple(row[1]) for row in self.pegs]))
        return tuple(self.state)

    def get_all_actions(self):
        all_actions = []
        for disc in self.discs:
            for peg in self.pegs:
                all_actions.append((disc, peg[0]))
        return all_actions

    def get_all_possible_actions(self):
        all_possible_actions = []
        if self.done_actions_count == 3:
            # to avoid repeating the same actions multiple times
            self.done_actions = []
            self.done_actions_count = 0
        for disc in self.discs:
            if disc in self.top_discs:
                for peg in self.pegs:
                    if (disc, peg[0]) not in self.done_actions:
                        if len(peg[1]) == 0:
                            # the peg is empty
                            all_possible_actions.append((disc, peg[0]))
                        elif peg[1][-1] > disc:
                            # the top disc is bigger than this disc
                            all_possible_actions.append((disc, peg[0]))
        return all_possible_actions

    def do_action(self, disc_newPeg):
        # The action is [the disc to move, the new peg]
        reward = 0
        self.time_step_count += 1
        if self.display:
            print('time_step_count', self.time_step_count)

        this_disc = disc_newPeg[0]
        new_peg = disc_newPeg[1]
        self.ac.append(disc_newPeg) # append it to make an action serie for each game
        if self.display:
            print('this_disc', this_disc, 'new_peg', new_peg)
        for peg in self.pegs:
            if this_disc in peg[1]:
                old_peg = peg[0] # get the old peg where the disc was
                break

        disc_to_remove_from_top = self.pegs[old_peg][1][-1]
        self.pegs[old_peg][1] = self.pegs[old_peg][1][:-1]
        if len(self.pegs[new_peg][1]) > 0:
            disc_to_remove_from_top2 = self.pegs[new_peg][1][-1] # remove the top disc of the new peg from top lists
            if disc_to_remove_from_top2 in self.top_discs:
                self.top_discs.remove(disc_to_remove_from_top2)
        self.pegs[new_peg][1].append(this_disc) # Add this disc to the new peg

        if len(self.top_discs) > 0 and disc_to_remove_from_top in self.top_discs:
            self.top_discs.remove(disc_to_remove_from_top) # remove this disc from the top list, but will be re-appended later
        self.top_discs.append(this_disc)

        if len(self.pegs[old_peg][1]) > 0:
            self.top_discs.append(self.pegs[old_peg][1][-1]) # added to top list again
        reward -= 1.4 # each step gives - reward

        if len(self.pegs[new_peg][1]) == self.discs_count and new_peg != self.init_peg:
            self.terminated = True # all the discs are moved to a new peg

        self.state[this_disc] = (new_peg, len(self.pegs[new_peg][1]) - 1) # update the state of this disc with new peg and new order

        self.done_actions.append(disc_newPeg)
        self.done_actions_count += 1

        self.top_discs = list(set(self.top_discs))
        self.one_game += (tuple([tuple(row[1]) for row in self.pegs])) # add this step to the game

        if self.time_step_count > self.max_T: # max time steps
            reward -= 20
            self.terminated = True

        if self.terminated:
            self.all_games.append(self.one_game) # add this game to all games, will be used in the plot part
            self.one_game = ()
            self.act.append(self.ac)
            self.ac = []
            if len(self.all_time_steps) > 0:
                if self.time_step_count < min(self.all_time_steps):
                    reward += 110
            self.all_time_steps.append(self.time_step_count)
            if self.time_step_count < 20:
                reward += 100
            elif self.time_step_count < 40:
                reward += 110
            elif self.time_step_count < 50:
                reward += 70
            elif self.time_step_count < 100:
                reward -= 20
            elif self.time_step_count < 150:
                reward -= 10
            else:
                reward -= 5

        return reward


    def get_state(self):
        return tuple(self.state)

    def is_terminated(self):
        return self.terminated

    def get_all_time_steps(self):
        return self.all_time_steps

    def get_all_state_counts(self):
        return self.discs_count**2 * self.pegs_count

    def get_all_action_count(self):
        return len(self.get_all_actions())

    def get_all_sap_count(self):
        return self.get_all_action_count() * self.get_all_state_counts()

    def get_all_states(self):
        all_states = []
        all_peg_orders = []
        for peg in range(self.pegs_count):
            for order in range(len(self.discs)):
                all_peg_orders.append([peg, order])
        return all_states # It returns an empty list

    def plot_timesteps_episodes(self, episods):
        # plot the time steps with the episodes
        fig, ax = plt.subplots()
        plt.plot(range(episods), self.get_all_time_steps())
        ax.set_xlabel('episodes')
        ax.set_ylabel('Time steps')
        plt.show()

    def plot_best_game(self):
        # get the best game and plot its steps
        min_steps = min(self.all_time_steps)
        print(min_steps)
        mix_index = self.all_time_steps.index(min_steps)
        best_game = self.all_games[mix_index]
        for game_index in range(0, len(best_game), self.pegs_count):
            fig, ax = plt.subplots()
            for peg_i in range(self.pegs_count):
                for d_i, disc in enumerate(best_game[game_index+peg_i]):
                    Drawing_colored_circle = plt.Circle(((peg_i+1.8)/10, (d_i+1)/10), ((disc+1)/100)+0.04)
                    ax.set_aspect(1)
                    ax.add_artist(Drawing_colored_circle)
                    plt.title('Colored Circle')
            plt.xlim(0, (self.pegs_count+2)/10)
            ax.set_xlabel('Pegs')
            ax.set_ylabel('Discs')
            plt.show()



