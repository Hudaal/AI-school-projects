
class Agent:
    """The agent which with use the actor-critic system"""
    def __init__(self, sim_world, actor, critic, critic_table=True) -> None:
        self.sim_world = sim_world
        self.actor = actor
        self.critic = critic
        self.critic_table = critic_table

    def train(self, episode_count, display=False):
        """Trains the agent on number of episodes"""
        # init the variables
        all_episods_score = []
        all_states = self.sim_world.get_all_states()
        self.actor.init_policy(self.sim_world, all_states)
        if self.critic_table:
            self.critic.init_values(self.sim_world, all_states)

        for episode in range(episode_count):
            # each episode reset the simWorld
            self.sim_world.reset()
            self.actor.init_elig(self.sim_world, all_states)
            if self.critic_table:
                self.critic.init_elig(self.sim_world, all_states)
            current_state = self.sim_world.get_init_state()
            current_action = self.actor.get_best_policy(current_state, self.sim_world, all_states) #getting the best action
            sap = [] # list of all state-action pairs reached
            terminated_state = False
            score = 0
            visited_states = []

            while not terminated_state: # the episode on
                reward = self.sim_world.do_action(current_action)
                if display:
                    print('episode', episode, 'reward', reward, self.sim_world.is_terminated())
                next_state = self.sim_world.get_state() # after the action is done, get reward

                if not self.sim_world.is_terminated():
                    next_action = self.actor.get_best_policy(next_state, self.sim_world, all_states)
                else: # the game is terminated and no need for more actions
                    next_action = 0
                if display:
                    print('current state', current_state, 'current_action', current_action)
                self.actor.e[(current_state, current_action)] = 1

                target = reward + self.critic.discount_f * self.critic.get_state_value(next_state)
                delta = target - self.critic.get_state_value(current_state)

                if self.critic_table:
                    self.critic.e[current_state] = 1

                sap_set = list(set(sap))
                for i in range(len(sap)):
                    # go back to all applied actions and update eligibilities and policy in actor
                    state = sap[i][0]
                    act = sap[i][1]
                    if self.critic_table:
                        self.critic.update_state_eligibility(state, delta)
                    self.actor.update_target_policy(state, act, delta)
                    self.actor.update_eligibility(state, act)

                score += reward

                if self.sim_world.is_terminated():
                    terminated_state = True
                    if display:
                        print('score: ', score)
                    all_episods_score.append(score)
                    self.actor.epsilon_greedy *= self.actor.epsilon_greedy_decay

                else:
                    if not self.critic_table:
                        # if the NN critic is used
                        self.critic.fit_nn(current_state, target)
                # if current_state not in visited_states:
                sap.append((current_state, current_action)) # done state and action
                visited_states.append(current_state)
                current_state = next_state # move to next state and action
                current_action = next_action

        return all_episods_score
            

            


