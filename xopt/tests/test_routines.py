from ..routines.batched import Batched
from ..evaluators.evaluator import Evaluator
from ..algorithms.random import RandomSample
from .test_functions.quad_3d import VOCS, evaluate
from ..tools import DummyExecutor


class TestRoutines:
    def test_batched(self):
        E = Evaluator(evaluate,
                      DummyExecutor(),
                      VOCS)

        alg = RandomSample(VOCS, 3, 2)
        config = {'vocs': VOCS}
        r = Batched(config, E, alg)
        print(r.run())

