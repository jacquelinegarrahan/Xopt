import logging
from copy import deepcopy

import yaml

from xopt import __version__
from xopt.legacy import reformat_config
from xopt.tools import expand_paths, load_config, get_function, isotime
from .configure import fill_defaults, ALL_DEFAULTS, configure_vocs
from .tools import DummyExecutor, get_n_required_fuction_arguments, \
    get_function_defaults

logger = logging.getLogger(__name__)

from .generators.generator import FunctionalGenerator
from .generators import KNOWN_ALGORITHMS
from .evaluators.evaluator import Evaluator
from .algorithms import KNOWN_ROUTINES


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

        self.routine = None
        self.generator = None
        self.evaluator = None
        self.vocs = None

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
        # make sure config dict has the required keys
        for name in ALL_DEFAULTS:
            if name not in self.config:
                raise Exception(f'Key {name} is required in config for Xopt')

        # reformat old config files if needed
        self.config = reformat_config(self.config)

        # load any high level config files
        for ele in ['xopt', 'evaluate', 'generator', 'vocs']:
            self.config[ele] = load_config(self.config[ele])

        # expand all paths
        self.config = expand_paths(self.config, ensure_exists=True)

        self.configure_vocs()
        self.configure_generator()
        self.configure_evaluate()
        self.configure_generator()

        self._configured_dict = self.config

    # --------------------------
    # Configures
    def configure_algorithm(self):
        """ configure algorithm """
        assert self.generator and self.evaluator, 'generator and evaluator not ' \
                                                  'initialized yet!'
        rtype = self.config['xopt'].get('routine', 'batched')
        routine_options = self.config['xopt'].get('options', {})

        # get default options
        if rtype in KNOWN_ROUTINES:
            routine_default_options = get_function_defaults(KNOWN_ROUTINES[rtype])
            self.config['xopt']['options'] = fill_defaults(routine_options,
                                                           routine_default_options)
            self.routine = KNOWN_ROUTINES[rtype](self.config,
                                                 self.evaluator,
                                                 self.generator,
                                                 **self.config['xopt']['options']
                                                 )
        else:
            ValueError('must use a named routine')

    def configure_generator(self):
        """ configure generator """
        # get generator via name or function
        alg_name = self.config['generator'].get('name', None)
        alg_function = self.config['generator'].get('function', None)
        alg_options = self.config['generator'].get('options', {})

        if alg_name is not None:
            if alg_name in KNOWN_ALGORITHMS:
                # get generator object
                self.algorithm = KNOWN_ALGORITHMS[alg_name](self.vocs, **alg_options)
            else:
                raise ValueError(f'Name `{alg_name}` not in list of known '
                                 f'algorithms, {KNOWN_ALGORITHMS}')

        elif alg_function is not None:
            # create generator object from callable function
            alg_function = get_function(alg_function)
            self.algorithm = FunctionalGenerator(self.vocs,
                                                 alg_function,
                                                 **alg_options)
        else:
            raise ValueError('must use a named generator or specify a generator '
                             'function')

    def configure_evaluate(self):
        # check to make sure there are no extraneous keys
        for key in self.config['evaluate'].keys():
            if key not in ['function', 'executor', 'options', 'name']:
                raise KeyError(f'key: {key} not allowed in evaluate config')

        evaluate_function = get_function(self.config['evaluate']['function'])
        evaluate_function_defaults = get_function_defaults(evaluate_function)

        executor = self.config['evaluate'].get('executor', DummyExecutor())
        evaluate_options = self.config['evaluate'].get('options', {}) or {}
        self.config['evaluate']['options'] = fill_defaults(evaluate_options,
                                                           evaluate_function_defaults)

        # check evaluate_function
        n_required_args = get_n_required_fuction_arguments(evaluate_function)
        if n_required_args != 1:
            raise ValueError(f'function has {n_required_args}, but should have '
                             f'exactly one. ')

        self.evaluator = Evaluator(self.vocs,
                                   evaluate_function,
                                   executor,
                                   evaluate_options)

    def configure_vocs(self):
        self.config['vocs'] = configure_vocs(self.config['vocs'])
        self.vocs = self.config['vocs']

    # --------------------------
    # Loading from file
    def load(self, config):
        """Load config from file (JSON or YAML) or data"""
        self.config = load_config(config)
        self.configure_all()

    @property
    def evaluate_config(self):
        return self.config['evaluate']

    @property
    def algorithm_config(self):
        return self.config['generator']

    # --------------------------
    # Run
    def run(self, executor=None):
        # check to make sure that configured_dict is equal to current config,
        # otherwise re-run configure_all
        if self._configured_dict != self.config:
            logger.debug('configured_dict != current config, redoing configuration')
            self.configure_all()

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
Configured: {self._configured_dict == self.config}
Config as YAML:
"""
        # return s+pprint.pformat(self.config)
        return s + yaml.dump(self.config, default_flow_style=None,
                             sort_keys=False)

    def __str__(self):
        return self.__repr__()
