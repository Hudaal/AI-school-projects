import math
import os

import matplotlib
matplotlib.use('Agg')
from numpy import sin, cos
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
import pandas as pd


class Display:

    """Class to represent the visualizing of the animation"""

    def __init__(self, l_1=1, l_2=1, m_1=1, m_2=1, speed=1, figsize=(7, 7)):
        self.graph = nx.path_graph(3)
        self.l_1 = l_1
        self.l_2 = l_2
        self.m_1 = m_1
        self.m_2 = m_2
        self.frame_delay = math.ceil(1000/(24 * speed))
        self.figsize = figsize
        self.dim = (l_1+l_2)*1.1

    def animate_episode(self, filepath, savepath):
        """Main method used to animate the graph"""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.clear()
        ax.grid()
        states = self.get_states(filepath)
        state = states[0]
        nodepos = self.compute_position(state)
        node_sizes = [self.m_1*200, self.m_2*200, 0]
        nx.draw(self.graph, node_color='#3876AF', edgecolors='#000000',
                edge_color='#3876AF', width=6, pos=nodepos, node_size=node_sizes)
        ani = animation.FuncAnimation(fig, self.update, frames=(range(1, len(states))), fargs=(
            ax, self, nodepos, states), interval=self.frame_delay, repeat=False, blit=False, save_count=len(states))
        ax.set_xlim([-self.dim, self.dim])
        ax.set_ylim([-self.dim, self.dim])

        if savepath is not None:
            writer = animation.PillowWriter(fps=30)
            ani.save(savepath, writer=writer, progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
        #plt.show()

    def update(self, i, ax, obj, nodepos, states):
        """Updates each frame of the animation"""
        ax.clear()
        ax.grid()
        ax.set_title(f'Control action {i // 4}')
        ax.axhline(y=self.l_2)
        state = states[i]
        nodepos = self.compute_position(state)
        node_sizes = [self.m_1*200, self.m_2*200, 0]
        nx.draw(obj.graph, node_color='#3876AF', edgecolors='#000000',
                edge_color='#3876AF', width=6, pos=nodepos, node_size=node_sizes)
        ax.set_xlim([-self.dim, self.dim])
        ax.set_ylim([-self.dim, self.dim])
        return ax,

    def compute_position(self, theta):
        """Calculates the position of each of the nodes"""
        theta3 = sum(theta)
        x_1 = self.l_1*sin(theta[0])
        y_1 = -self.l_1*cos(theta[0])
        x_2 = x_1 + self.l_2*sin(theta3)
        y_2 = y_1 - self.l_2*cos(theta3)
        return [(0, 0), (x_1, y_1), (x_2, y_2)]

    def get_states(self, filepath):
        """Reads data from a csv file and returns
        an array of all the data given as states"""
        states = []
        df = pd.read_csv(filepath, delimiter=';')
        for _, row in df.iterrows():
            states.append((row["theta1"], row["theta2"]))
        return states


if __name__ == '__main__':
    L1 = 1.0
    L2 = 1.0
    M1 = 1.0
    M2 = 1.0

    # dir = 'output/220504-111146'
    # l = 1000000
    # f = None
    # for file in os.listdir(dir):
    #     if file.endswith('.png'):
    #         continue
    #     df = pd.read_csv(os.path.join(dir, file))
    #     if df.size < l:
    #         l = df.size
    #         f = file
    #
    # print(f, l)

    filepath = "output/220504-111146/episode431.csv"

    display = Display(L1, L2, M1, M2, speed=50)
    display.animate_episode(filepath, 'vids/episode431.gif')
