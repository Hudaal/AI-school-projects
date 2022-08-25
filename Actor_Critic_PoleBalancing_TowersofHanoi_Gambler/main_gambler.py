from Agent import Agent
from Actor import Actor
from Critic_table_based import Critic_table_based
from Critic_NN_based import Critic_NN_based
from Gambler_SimWorld import Gambler_SimWorld
import matplotlib.pyplot as plt


def main():
    env = Gambler_SimWorld(0.4)
    actor = Actor(0.03, 0.4, 0.4, 0.1, 0.001)
    # critic = Critic_table_based(0.03, 0.4, 0.4)
    critic = Critic_NN_based(0.01, 0.9, 0.9, 1, 1, 2, neurons=[4, 8])
    agent = Agent(env, actor, critic, False)
    agent.train(500)

    env.plot_state_best_action()

main()