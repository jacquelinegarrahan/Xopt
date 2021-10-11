from .random import RandomSample
KNOWN_ALGORITHMS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sample': RandomSample,
    'bayesian_optimization': 'xopt.bayesian.algorithms.bayesian_optimize',
    'bayesian_exploration': 'xopt.bayesian.algorithms.bayesian_exploration',
    'mobo': 'xopt.bayesian.algorithms.mobo',
    'multi_fidelity': 'xopt.bayesian.algorithms.multi_fidelity_optimize'
}