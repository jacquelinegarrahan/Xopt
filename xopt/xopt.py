import logging
from copy import deepcopy

import yaml

from xopt import __version__
from xopt.legacy import reformat_config
from xopt.tools import expand_paths, load_config, get_function, isotime
from .configure import ALL_DEFAULTS, VOCS_DEFAULTS, EVALUATE_DEFAULTS, \
    ALGORITHM_DEFAULTS
from .tools import DummyExecutor, get_n_required_fuction_arguments, \
    get_function_defaults
from .utils import check_and_fill_defaults

logger = logging.getLogger(__name__)

from .generators.generator import FunctionalGenerator
from .generators import KNOWN_GENERATORS
from .evaluators.evaluator import Evaluator
from .algorithms import KNOWN_ALGORITHMS


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

        self.generator = None
        self.algorithm = None
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

        # check high level options are correct for algorithm
        algorithm_config = check_and_fill_defaults(
            self.config['algorithm'],
            ALGORITHM_DEFAULTS
        )

        # get default algorithm options
        algorithm_type = algorithm_config['type']
        algorithm_default_options = {}
        if algorithm_type in KNOWN_ALGORITHMS:
            algorithm_default_options = get_function_defaults(
                KNOWN_ALGORITHMS[algorithm_type])
        else:
            ValueError('must use a named algorithm')

        # get generator function/object
        generator_type = algorithm_config['name']
        if algorithm_config['function'] is not None:
            generator_function = get_function(algorithm_config['function'])
        else:
            generator_function = None

        # get generator defaults
        if generator_type is not None and generator_function is None:

            # case where generator is specified by name
            if generator_type in KNOWN_GENERATORS:
                generator_default_options = get_function_defaults(
                    KNOWN_GENERATORS[generator_type]
                )
            else:
                raise ValueError('unknown generator name specified')
        elif generator_type is None and generator_function is not None:

            # case where no generator is specified but function is given by name
            function_default_options = get_function_defaults(generator_function)
            generator_default_options = get_function_defaults(FunctionalGenerator)
            generator_default_options.update(function_default_options)
        else:
            # case where there is a problem
            raise ValueError('Either `name` or `function` must be specified')

        # combine defaults dict for the generator and the algorithm into one
        all_default_options = {**algorithm_default_options,
                               **generator_default_options}

        # get options from config file
        algorithm_config['options'] = check_and_fill_defaults(
            algorithm_config['options'] or {}, all_default_options)

        # create generator object
        generator_kwargs = {k: algorithm_config['options'][k] for k in
                            generator_default_options.keys()}
        if generator_type is not None:
            self.generator = KNOWN_GENERATORS[generator_type](self.config['vocs'],
                                                              **generator_kwargs)
        else:
            self.generator = FunctionalGenerator(self.config['vocs'],
                                                 generator_function,
                                                 **generator_kwargs)

        # create algorithm object
        algorithm_kwargs = {k: algorithm_config['options'][k] for k in
                            algorithm_default_options.keys()}
        self.algorithm = KNOWN_ALGORITHMS[algorithm_type](
            self.config,
            self.evaluator,
            self.generator,
            **algorithm_kwargs)

        # update config object
        self.config['algorithm'] = algorithm_config

    def configure_evaluate(self):
        # check high level options are correct for evaluate
        evaluate_config = check_and_fill_defaults(
            self.config['evaluate'],
            EVALUATE_DEFAULTS
        )

        evaluate_function = get_function(evaluate_config['function'])
        evaluate_function_defaults = get_function_defaults(evaluate_function)

        evaluate_config['options'] = check_and_fill_defaults(
            evaluate_config['options'] or {},
            evaluate_function_defaults)

        executor = evaluate_config['executor'] or DummyExecutor()
        evaluate_options = evaluate_config['options']

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
        self.config['vocs'] = check_and_fill_defaults(self.config['vocs'],
                                                      VOCS_DEFAULTS)
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
