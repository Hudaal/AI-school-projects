from Agent import Agent
from Actor import Actor
from Critic_table_based import Critic_table_based
from Critic_NN_based import Critic_NN_based
from TowersofHanoiSimWorld import TowersofHanoi_SimWorld
import matplotlib.pyplot as plt


def main():
    env = TowersofHanoi_SimWorld()
    actor = Actor(0.09, 0.7, 0.7, 0.001, 0.00001)
    critic = Critic_table_based(0.001, 0.7, 0.7)
    # critic = Critic_NN_based(0.0001, 0.7, 0.7, 4, 1, 2, neurons=[4, 2])
    agent = Agent(env, actor, critic, True)
    episodes = 201
    agent.train(episodes)

    env.plot_timesteps_episodes(episodes)
    env.plot_best_game()

main()