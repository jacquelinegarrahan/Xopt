"""
Tools to configure an xopt run

"""
from copy import deepcopy

from .evaluators import DummyExecutor
import logging
from typing import Dict

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
    'type': 'batched',
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



