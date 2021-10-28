from xopt.algorithms.generators import RandomSample
from .test_functions.TNK import VOCS


class TestRandom:
    def test_random(self):
        rand_gen = RandomSample(VOCS)
        rand_gen.generate(None)
