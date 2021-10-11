import logging
from copy import deepcopy

import yaml

from xopt import __version__
from xopt.legacy import reformat_config
from xopt.tools import expand_paths, load_config, get_function, isotime
from . import configure
from .tools import DummyExecutor

logger = logging.getLogger(__name__)

from .algorithms.algorithm import FunctionalAlgorithm
from .algorithms import KNOWN_ALGORITHMS
from .evaluators.evaluator import Evaluator
from .routines import KNOWN_ROUTINES


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
        self.config = deepcopy(config)
        self.configured = False

        self.routine = None
        self.algorithm = None
        self.evaluator = None
        self.vocs = None

        if config is not None:
            self.config = load_config(self.config)

            # make sure configure has the required keys
            for name in configure.ALL_DEFAULTS:
                if name not in self.config:
                    raise Exception(f'Key {name} is required in config for Xopt')

            # reformat old config files if needed
            self.config = reformat_config(self.config)

            # load any high level config files
            for ele in ['xopt', 'evaluate', 'algorithm', 'vocs']:
                self.config[ele] = load_config(self.config[ele])

            # do configuration
            self.configure_all()

        else:
            # Make a template, so the user knows what is available
            logger.info('Initializing with defaults')
            self.config = deepcopy(configure.ALL_DEFAULTS)

    def configure_all(self):
        """
        Configure everything

        Configuration order:
        xopt
        algorithm
        simulation
        vocs, which contains the simulation name, and templates

        """
        # expand all paths
        self.config = expand_paths(self.config, ensure_exists=True)

        self.configure_vocs()
        self.configure_algorithm()
        self.configure_evaluate()
        self.configure_routine()

        self.configured = True

    # --------------------------
    # Configures
    def configure_routine(self):
        """ configure routine """
        assert self.algorithm and self.evaluator, 'algorithm and evaluator not ' \
                                                  'initialized yet!'
        rtype = self.config['xopt'].get('routine', 'batched')
        path = self.config['xopt'].get('output_path', '.')
        if rtype in KNOWN_ROUTINES:
            self.routine = KNOWN_ROUTINES['rtype'](self.config,
                                                   self.evaluator,
                                                   self.algorithm,
                                                   output_path=path
                                                   )
        else:
            ValueError('must use a named routine')

    def configure_algorithm(self):
        """ configure algorithm """
        # get algorithm via name or function
        alg_name = self.config['algorithm'].get('name', None)
        alg_function = self.config['algorithm'].get('function', None)
        alg_options = self.config['algorithm'].get('options', {})

        if alg_name is not None:
            if alg_name in KNOWN_ALGORITHMS:
                # get algorithm object
                self.algorithm = KNOWN_ALGORITHMS[alg_name](self.vocs, **alg_options)
            else:
                raise ValueError(f'Name `{alg_name}` not in list of known '
                                 f'algorithms, {KNOWN_ALGORITHMS}')

        elif alg_function is not None:
            # create algorithm object from callable function
            alg_function = get_function(alg_function)
            self.algorithm = FunctionalAlgorithm(self.vocs,
                                                 alg_function,
                                                 **alg_options)
        else:
            raise ValueError('must use a named algorithm or specify a algorithm '
                             'function')

    def configure_evaluate(self):
        evaluate_function = get_function(self.config['evaluate']['function'])
        executor = self.config['evaluate'].get('executor', DummyExecutor())
        evaluate_options = self.config['evaluate'].get('options', {})
        self.evaluator = Evaluator(self.vocs,
                                   evaluate_function,
                                   executor,
                                   evaluate_options)

    def configure_vocs(self):
        self.config['vocs'] = configure.configure_vocs(self.config['vocs'])

    # --------------------------
    # Saving and Loading from file
    def load(self, config):
        """Load config from file (JSON or YAML) or data"""
        self.config = load_config(config)
        self.configure_all()

    # --------------------------
    # Run

    def run(self, executor=None):
        assert self.configured, 'Not configured to run.'

        logger.info(f'Starting at time {isotime()}')

        return self.routine.run()

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
Configured: {self.configured}
Config as YAML:
"""
        # return s+pprint.pformat(self.config)
        return s + yaml.dump(self.config, default_flow_style=None,
                             sort_keys=False)

    def __str__(self):
        return self.__repr__()
