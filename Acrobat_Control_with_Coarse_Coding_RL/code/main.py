import argparse
import os


from environments.environment import Environment
from learner.reinforcement_learner import ReinforcementLearner
from utils.config_parser import ConfigParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/config/file', default='configs/base_config.yaml')
parser.add_argument('-v', '--visualize', action='store_true', help='Flag used to get visualizations.')
args = parser.parse_args()

config_parser = ConfigParser(args.config)

environment: Environment = config_parser.environment

rl: ReinforcementLearner = config_parser.reinforcement_learner

fit_parameters = config_parser.fit_parameters
visualization_parameters = config_parser.visualization_parameters
show = visualization_parameters['show']
vis_sleep = visualization_parameters['vis_sleep']

print('---FITTING MODEL---')
rl.fit(**fit_parameters)
if show:
    rl.visualize_fit()

print('---RUNNING MODEL---')
rl.run(visualize=show, vis_sleep=vis_sleep)
