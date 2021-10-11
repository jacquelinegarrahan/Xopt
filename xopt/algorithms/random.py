from .algorithm import FunctionalAlgorithm
from ..vocs_tools import get_bounds
import numpy as np


class RandomSample(FunctionalAlgorithm):
    def __init__(self, vocs, n_samples=1, max_calls=1):
        """
        Algorithm to do random sampling
        """

        def _random_sample(vocs):
            bounds = get_bounds(vocs)
            r = np.random.rand(n_samples, bounds.shape[-1])
            return (r - bounds[0]) / (bounds[1] - bounds[0])

        super(RandomSample, self).__init__(vocs,
                                           _random_sample,
                                           {'max_calls': max_calls})
