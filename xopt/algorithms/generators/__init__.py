from .bayesian.base import BayesianGenerator
from .bayesian.generator import (
    UpperConfidenceBound,
    ExpectedHypervolumeImprovement,
    BayesianExploration,
    MultiFidelity,
    QualityAware,
)
from .random import RandomSample
from .generator import Generator, ContinuousGenerator, FunctionalGenerator
KNOWN_GENERATORS = {
    "random_sample": RandomSample,
    "upper_confidence_bound": UpperConfidenceBound,
    "bayesian_optimization": BayesianGenerator,
    "bayesian_exploration": BayesianExploration,
    "expected_hypervolume_improvement": ExpectedHypervolumeImprovement,
    "multi_fidelity": MultiFidelity,
    "quality_aware_exploration": QualityAware,
}
