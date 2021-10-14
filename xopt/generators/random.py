import pandas as pd

from .generator import ContinuousGenerator
from ..vocs_tools import get_bounds
import numpy as np


class RandomSample(ContinuousGenerator):
    def is_terminated(self):
        return self.n_calls > self.max_calls

    def _generate(self, data) -> pd.DataFrame:
        bounds = get_bounds(self.vocs)
        r = np.random.rand(self._n_samples, bounds.shape[-1])
        result = r * (bounds[1] - bounds[0]) + bounds[0]
        return self.numpy_to_dataframe(result)

    def __init__(self, vocs, n_samples=1, max_calls=1):
        """
        Generator to do random sampling
        """
        super(RandomSample, self).__init__(vocs)
        self.max_calls = max_calls
        self.set_n_samples(n_samples)


