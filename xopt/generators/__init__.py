from .random import RandomSample
from .bayesian.generator import UpperConfidenceBound, ExpectedHypervolumeImprovement
KNOWN_GENERATORS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sample': RandomSample,
    'upper_confidence_bound': UpperConfidenceBound,
    'bayesian_optimization': 'xopt.bayesian.generators_old.bayesian_optimize',
    'bayesian_exploration': 'xopt.bayesian.generators_old.bayesian_exploration',
    'expected_hypervolume_improvement': ExpectedHypervolumeImprovement,
    'multi_fidelity': 'xopt.bayesian.generators_old.multi_fidelity_optimize'
}