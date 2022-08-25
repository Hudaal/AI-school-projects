import inspect

import yaml

from environments.environment import Environment
from environments.cartpole import CartPole
from environments.towers_of_hanoi import TowersOfHanoi
from environments.gambler import Gambler
from environments.acrobat import Acrobat
from learner.critics.network_critic import NetworkCritic
from learner.reinforcement_learner import ReinforcementLearner

STRING_EXCEPTIONS = ['name', 'checkpoint_folder']


class ConfigParser:
    def __init__(self, config_file: str):
        with open(config_file, "r") as stream:
            self._config = yaml.safe_load(stream)

        self.environment: Environment = self._get_environment()
        self.reinforcement_learner: ReinforcementLearner = self._get_reinforcement_learner()
        self.fit_parameters: dict = self._get_fit_parameters()
        self.visualization_parameters: dict = self._get_visualization_parameters()

    def _parse_config(self, config) -> dict:
        parsed_config = {}
        for k, v in config.items():
            if v is not None:
                if type(v) is dict:
                    parsed_config[k] = self._parse_config(v)
                elif type(v) is str and k not in STRING_EXCEPTIONS:
                    parsed_config[k] = eval(v)
                else:
                    parsed_config[k] = v
        return parsed_config

    def _get_environment(self) -> Environment:
        environment = eval(self._config['environment_type'])
        kwargs = self._parse_config(self._config['environment_params'])
        return environment(**kwargs)

    def _get_critic(self) -> NetworkCritic:
        critic = eval(self._config['critic_type'])
        kwargs = self._parse_config(self._config['critic_params'])
        return critic(environment=self.environment, **kwargs)

    def _get_reinforcement_learner(self) -> ReinforcementLearner:
        critic = self._get_critic()
        return ReinforcementLearner(environment=self.environment, critic=critic)

    def _get_fit_parameters(self) -> dict:
        return self._parse_config(self._config['fit'])

    def _get_visualization_parameters(self) -> dict:
        return self._parse_config(self._config['visualization'])
