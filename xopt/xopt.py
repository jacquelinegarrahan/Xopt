import logging
from copy import deepcopy

import yaml

from xopt import __version__
from xopt.legacy import reformat_config
from xopt.tools import expand_paths, load_config, isotime
from .configure import ALL_DEFAULTS, VOCS_DEFAULTS, parse_algorithm_config, \
    parse_evaluator_config
from .utils import check_and_fill_defaults

logger = logging.getLogger(__name__)

from .generators.generator import FunctionalGenerator
from .generators import KNOWN_GENERATORS
from .algorithms.algorithm import Algorithm
from .evaluators.evaluator import Evaluator


class Xopt:
    """
    
    Object to handle a single optimization problem.
    
    Parameters
    ----------
    config: dict, YAML text, JSON text
        input file should be a dict, JSON, or YAML file with top level keys
    
          
    """

    def __init__(self, config=None):

        # Internal state

        # Main configuration is in this nested dict
        config = deepcopy(config)
        self._configured_dict = None

        self._algorithm = None
        self._evaluator = None
        self._vocs = None

        if config is not None:
            config = load_config(config)

            # set configuration and configure all of the objects
            self.config = config
            self.configure_all()

        else:
            # Make a template, so the user knows what is available
            logger.info('Initializing with defaults')
            self.config = deepcopy(ALL_DEFAULTS)

    def configure_all(self):
        """
        Configure everything

        Configuration order:
        xopt
        generator
        simulation
        vocs, which contains the simulation name, and templates

        """
        # reformat old config files if needed
        self.config = reformat_config(self.config)

        # make sure config dict has the required keys
        for name in ALL_DEFAULTS:
            if name not in self.config:
                raise Exception(f'Key {name} is required in config for Xopt')

        # load any high level config files
        for ele in ['xopt', 'evaluate', 'algorithm', 'vocs']:
            self.config[ele] = load_config(self.config[ele])

        # expand all paths
        self.config = expand_paths(self.config, ensure_exists=True)

        self.configure_vocs()
        self.configure_evaluate()
        self.configure_algorithm()

        self._configured_dict = deepcopy(self.config)

    # --------------------------
    # Configures
    def configure_algorithm(self):
        """ configure generator and algorithm """
        # parse config dict
        algorithm_config, algorithm_kwargs, \
            generator_config, generator_kwargs = \
            parse_algorithm_config(self.config['algorithm'])

        vocs = self.config['vocs']

        # create generator object
        if generator_config['type'] is not None:
            generator = KNOWN_GENERATORS[generator_config['type']](
                vocs, **generator_kwargs)
        else:
            generator = FunctionalGenerator(self.config['vocs'],
                                            generator_config['generator_function'],
                                            **generator_kwargs)

        # create algorithm object
        self._algorithm = Algorithm(
            self.config['vocs'],
            self._evaluator,
            generator,
            **algorithm_kwargs)

        # update config dict
        self.config['algorithm'] = algorithm_config

    def configure_evaluate(self):
        # parse evaluate dict
        evaluate_function, executor, evaluate_options = parse_evaluator_config(
            self.config['evaluate'])

        # create evaluator object
        self._evaluator = Evaluator(self._vocs,
                                    evaluate_function,
                                    executor,
                                    evaluate_options)

    def configure_vocs(self):
        self.config['vocs'] = check_and_fill_defaults(self.config['vocs'],
                                                      VOCS_DEFAULTS)
        self._vocs = self.config['vocs']

    # --------------------------
    # Loading from file
    def load(self, config):
        """Load config from file (JSON or YAML) or data"""
        self.config = load_config(config)
        self.configure_all()

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def generator(self):
        return self._algorithm.generator

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def vocs(self):
        return self._vocs

    @property
    def evaluate_config(self):
        return self.config['evaluate']

    @property
    def algorithm_config(self):
        return self.config['algorithm']

    # --------------------------
    # Run
    def run(self):
        # check to make sure that configured_dict is equal to current config,
        # otherwise re-run configure_all
        if self._configured_dict != self.config:
            logger.debug('configured_dict != current config, redoing configuration')
            self.configure_all()

        logger.info(f'Starting at time {isotime()}')

        return self.algorithm.run()

    def __getitem__(self, config_item):
        """
        Get a configuration attribute
        """
        return self.config[config_item]

    def __repr__(self):
        s = f"""
            Xopt 
________________________________           
Version: {__version__}
Configured: {self._configured_dict == self.config}
Config as YAML:
"""
        # return s+pprint.pformat(self.config)
        return s + yaml.dump(self.config, default_flow_style=None,
                             sort_keys=False)

    def __str__(self):
        return self.__repr__()
