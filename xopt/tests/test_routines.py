from ..algorithms.batched import Batched
from ..evaluators.evaluator import Evaluator
from ..generators.random import RandomSample
from .test_functions.TNK import VOCS, evaluate_TNK
from ..tools import DummyExecutor


class TestRoutines:
    def test_batched(self):
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      DummyExecutor(),
                      )

        alg = RandomSample(VOCS, 3, 2)
        config = {'vocs': VOCS}
        r = Batched(config, E, alg)
        print(r.run())
