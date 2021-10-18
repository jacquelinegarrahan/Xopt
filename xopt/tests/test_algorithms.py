import json
from copy import deepcopy

import numpy as np
import pandas as pd

from .test_functions.TNK import VOCS, evaluate_TNK
from ..algorithms.batched import Batched
from ..algorithms.continuous import Continuous
from ..evaluators.evaluator import Evaluator
from ..generators.bayesian.generator import UpperConfidenceBound
from ..generators.random import RandomSample


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
        generator = UpperConfidenceBound(vocs, 2)

        # read in and load file
        alg = Batched(vocs, vocs, E, generator, n_initial_samples=5)
        alg.load_data(fname)
        alg.run()
        assert len(alg.data) == 17
