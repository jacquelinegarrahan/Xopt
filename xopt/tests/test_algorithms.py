from ..algorithms.batched import Batched
from ..algorithms.continuous import Continuous
from ..evaluators.evaluator import Evaluator
from ..generators.random import RandomSample
from .test_functions.TNK import VOCS, evaluate_TNK
from ..evaluators import DummyExecutor


class TestAlgorithms:
    def test_batched(self):
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      )

        gen = RandomSample(VOCS, 3, 2)
        config = {'vocs': VOCS}
        r = Batched(config, config['vocs'], E, gen)
        r.run()

    def test_continuous(self):
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      )

        gen = RandomSample(VOCS, 1, 5)
        config = {'vocs': VOCS}
        r = Continuous(config, config['vocs'], E, gen, 5)
        print(r.run())

