from .random import RandomSample
from .bayesian.algorithms import UpperConfidenceBound
KNOWN_ALGORITHMS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sample': RandomSample,
    'upper_confidence_bound': UpperConfidenceBound,
    'bayesian_optimization': 'xopt.bayesian.generators.bayesian_optimize',
    'bayesian_exploration': 'xopt.bayesian.generators.bayesian_exploration',
    'mobo': 'xopt.bayesian.generators.mobo',
    'multi_fidelity': 'xopt.bayesian.generators.multi_fidelity_optimize'
}