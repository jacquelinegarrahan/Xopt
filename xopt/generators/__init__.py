from .random import RandomSample
from .bayesian.generator import UpperConfidenceBound, \
    ExpectedHypervolumeImprovement, BayesianExploration, MultiFidelity, \
    QualityAware
from .bayesian.base import BayesianGenerator

KNOWN_GENERATORS = {
    'cnsga': 'xopt.cnsga.cnsga',
    'random_sample': RandomSample,
    'upper_confidence_bound': UpperConfidenceBound,
    'bayesian_optimization': BayesianGenerator,
    'bayesian_exploration': BayesianExploration,
    'expected_hypervolume_improvement': ExpectedHypervolumeImprovement,
    'multi_fidelity': MultiFidelity,
    'quality_aware_exploration': QualityAware
}