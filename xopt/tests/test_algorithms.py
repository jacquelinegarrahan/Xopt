import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from .test_functions.TNK import VOCS, evaluate_TNK
from ..evaluators.evaluator import Evaluator
from ..generators.bayesian.generator import BayesianExploration
from ..generators.random import RandomSample
from ..algorithms.algorithm import Algorithm


class TestAlgorithms:
    def test_batched(self):
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      )

        gen = RandomSample(VOCS, 3, 2)
        config = {'vocs': VOCS}
        r = Algorithm(config['vocs'], E, gen)
        r.run()

    def test_continuous(self):
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      )

        gen = RandomSample(VOCS, 1, 5)
        config = {'vocs': VOCS}
        r = Algorithm(config['vocs'], E, gen, n_initial_samples=5,
                      control_flow='continuous')
        print(r.run())

    def test_data_load(self):
        # generate some dummy data
        data = np.random.rand(10, 5)
        df = pd.DataFrame(data, columns=['x1', 'x2', 'y1', 'y2', 'c1'])

        fname = 'test_result.json'
        with open(fname, 'w') as outfile:
            json.dump({'results': df.to_dict()}, outfile)

        # run UCB with restart data
        E = Evaluator(VOCS,
                      evaluate_TNK,
                      )
        vocs = deepcopy(VOCS)
        del vocs['objectives']['y2']
        generator = BayesianExploration(vocs, 2, mc_samples=2)

        # read in and load file
        alg = Algorithm(vocs, E, generator, n_initial_samples=5)
        alg.load_data(fname)
        alg.run()
        assert len(alg.data) == 17
        os.remove(fname)

