import pandas as pd

from .generator import ContinuousGenerator
from ..vocs_tools import get_bounds
from botorch.utils.sampling import draw_sobol_samples
import torch


class RandomSample(ContinuousGenerator):
    def is_terminated(self):
        return self.n_calls > self.max_calls

    def _generate(self, data) -> pd.DataFrame:
        bounds = torch.tensor(get_bounds(self.vocs)).double()
        result = draw_sobol_samples(bounds, self._n_samples,
                                    1).numpy().reshape(
            self._n_samples, -1)
        return self.numpy_to_dataframe(result)

    def __init__(self, vocs, n_samples=1, max_calls=1):
        """
        Generator to do random sampling
        """
        super(RandomSample, self).__init__(vocs)
        self.max_calls = max_calls
        self.set_n_samples(n_samples)


