from ..evaluators.evaluator import Evaluator
from .test_functions.quad_3d import VOCS, evaluate
from ..tools import DummyExecutor
import numpy as np


class TestEvaluator:
    def test_evaluator(self):
        E = Evaluator(evaluate,
                      DummyExecutor(),
                      VOCS)

        samples = np.random.rand(3, 3)
        E.submit_samples(samples)
        print(f'\n{E.collect_results()}')
