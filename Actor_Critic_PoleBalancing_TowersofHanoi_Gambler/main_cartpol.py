from Agent import Agent
from Actor import Actor
from Critic_table_based import Critic_table_based
from Critic_NN_based import Critic_NN_based
from CartPol_SimWorld import cartPol_SimWorld
import matplotlib.pyplot as plt


def main():
    env = cartPol_SimWorld()
    actor = Actor(0.09, 0.5, 0.5, 0.01, 0.001)
    critic = Critic_table_based(0.001, 0.7, 0.7)
    # critic = Critic_NN_based(0.01, 0.9, 0.9, 2, 1, 2, neurons=[4, 8])
    agent = Agent(env, actor, critic, True)
    episods = 1800
    agent.train(episods)

    env.plot_best_timestep_angle()
    env.plot_timesteps_angles(episods)

main()