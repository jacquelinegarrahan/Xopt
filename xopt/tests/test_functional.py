from xopt.algorithms.functional import run_algortihm
from xopt.generators.bayesian.generator import BayesianExploration

# test function
from xopt.tests.test_functions import TNK


class TestFunctional:
    def test_functional(self):
        # Get VOCS
        vocs = TNK.VOCS

        # technically this is not necessary, but its good to be explict
        vocs['objectives'] = {'y1': None}

        # Get evaluate function
        EVALUATE = TNK.evaluate_TNK

        n_steps = 2

        # create generator object
        generator = BayesianExploration(vocs, n_steps)

        # Run - see comments for example options
        alg = run_algortihm(vocs, generator, function=EVALUATE)
