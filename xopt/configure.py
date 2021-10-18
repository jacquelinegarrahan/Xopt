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
    'options': None,
}

# Algorithms
ALGORITHM_DEFAULTS = {
    'name': None,
    'type': 'batched',
    'function': None,
    'options': None,
}

VOCS_DEFAULTS = {
    'variables': None,
    'objectives': None,
    'constraints': None,
    'linked_variables': None,
    'constants': None
}

ALL_DEFAULTS = {
    'xopt': XOPT_DEFAULTS,
    'algorithm': ALGORITHM_DEFAULTS,
    'evaluate': EVALUATE_DEFAULTS,
    'vocs': VOCS_DEFAULTS
}



