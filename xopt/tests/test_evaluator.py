import pandas as pd

from ..evaluators.evaluator import Evaluator
from .test_functions.quad_3d import VOCS, evaluate
from ..evaluators import DummyExecutor
import numpy as np


class TestEvaluator:
    def test_evaluator(self):
        E = Evaluator(VOCS,
                      evaluate,
                      )

        samples = np.random.rand(10, 3)
        samples = pd.DataFrame(samples, columns=VOCS['variables'])
        E.submit_samples(samples)
        print(f'\n{E.collect_results()}')
