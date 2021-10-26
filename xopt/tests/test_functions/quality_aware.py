import numpy as np
import torch
import time
from botorch.test_functions.multi_fidelity import AugmentedHartmann
import logging

logger = logging.getLogger(__name__)
VOCS = {
    'name': '2D quality test',
    'description': '2D quality test function',
    'variables': {
        'x1': [0, 1.0],
        'x2': [0, 1.0],
    },
    'objectives': {
        'y1': 'MINIMIZE',
        'q1': 'MAXIMIZE'

    },
    'constraints': {},
    'constants': {}
}


# labeled version
def evaluate(inputs, extra_option='abc', **params):
    x = np.array((inputs['x1'], inputs['x2']))
    z = x[0] - x[1]
    outputs = {'y1': (x[0] - 0.5)**2,
               'q1': np.exp(-(z - 0.5))*np.sin(z + 0.5)}

    return outputs

