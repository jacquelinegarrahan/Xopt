from .test_functions.TNK import VOCS
from ..generators.random import RandomSample

class TestRandom:
    def test_random(self):
        rand_gen = RandomSample(VOCS)
        rand_gen.generate(None)