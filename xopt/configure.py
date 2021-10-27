"""
Tools to configure an xopt run

"""
from copy import deepcopy

import logging
from typing import Dict
from .tools import get_function_defaults, get_n_required_fuction_arguments, get_function
from .utils import check_and_fill_defaults
from .generators.generator import FunctionalGenerator
from .generators import KNOWN_GENERATORS
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .algorithms.algorithm import Algorithm

logger = logging.getLogger(__name__)

# defaults for required dict keys
XOPT_DEFAULTS = {
    'output_path': '.'
}

EVALUATE_DEFAULTS = {
    'name': None,
    'function': None,
    'executor': None,
    'options': {},
}

# Algorithms
ALGORITHM_DEFAULTS = {
    'name': None,
    'function': None,
    'options': {},
}

VOCS_DEFAULTS = {
    'variables': {},
    'objectives': {},
    'constraints': {},
    'linked_variables': {},
    'constants': {}
}

ALL_DEFAULTS = {
    'xopt': XOPT_DEFAULTS,
    'algorithm': ALGORITHM_DEFAULTS,
    'evaluate': EVALUATE_DEFAULTS,
    'vocs': VOCS_DEFAULTS
}


def parse_algorithm_config(algorithm_config: Dict):
    # check high level options are correct for algorithm
    algorithm_config = check_and_fill_defaults(
        algorithm_config,
        ALGORITHM_DEFAULTS
    )

    # get default algorithm options

    algorithm_default_options = get_function_defaults(Algorithm)

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

    generator_config = {'type': generator_type,
                        'generator_function': generator_function}

    # create generator object
    generator_kwargs = {k: algorithm_config['options'][k] for k in
                        generator_default_options.keys()}

    algorithm_kwargs = {k: algorithm_config['options'][k] for k in
                        algorithm_default_options.keys()}

    return algorithm_config, algorithm_kwargs, generator_config, \
           generator_kwargs


def parse_evaluator_config(evaluator_config: Dict):
    # check high level options are correct for evaluate
    evaluator_config = check_and_fill_defaults(
        evaluator_config,
        EVALUATE_DEFAULTS
    )

    evaluate_function = get_function(evaluator_config['function'])
    evaluate_function_defaults = get_function_defaults(evaluate_function)

    evaluator_config['options'] = check_and_fill_defaults(
        evaluator_config['options'] or {},
        evaluate_function_defaults)

    executor = evaluator_config['executor']

    # parse executor if string
    if isinstance(executor, str):
        if executor == 'ThreadPoolExecutor':
            executor = ThreadPoolExecutor()
        elif executor == 'ProcessPoolExecutor':
            executor = ProcessPoolExecutor()
        else:
            raise RuntimeError(f'specified executor via string {executor} does not '
                               f'exit')

    evaluate_options = evaluator_config['options']

    # check evaluate_function
    n_required_args = get_n_required_fuction_arguments(evaluate_function)
    if n_required_args != 1:
        raise ValueError(f'function has {n_required_args}, but should have '
                         f'exactly one. ')

    return evaluate_function, executor, evaluate_options
