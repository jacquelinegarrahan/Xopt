from ..algorithm import FunctionalAlgorithm
from typing import Dict
import numpy as np


class CNSGA(FunctionalAlgorithm):
    def __init__(self, vocs: Dict, n_generations=1, **kwargs):
        options = kwargs
        options['max_calls'] = n_generations
        super(CNSGA, self).__init__(vocs, self.generate_population, options)

    def generate_population(self, X, Y, **kwargs) -> np.ndarray:
        """
        fill in code here for CNSGA generation generation
        """


