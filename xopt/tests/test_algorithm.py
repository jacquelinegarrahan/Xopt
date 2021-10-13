from ..generators.generator import Generator, FunctionalGenerator, BadFunctionError
import numpy as np
import pandas as pd
import pytest


class TestAlgorithmBase:
    vocs = {'variables':
                {'x1': [0, 1],
                 'x2': [0, 1]},
            'objectives':
                {'y1': 'MINIMIZE'}}

    def test_algorithm_base(self):
        def a(vocs):
            return np.random.rand(5, 2)

        def b(vocs, x=None):
            return np.random.rand(5, 2)

        def c(vocs, data):
            return np.random.rand(5, 2)

        def d(vocs, data, x=None):
            return np.random.rand(5, 2)

        def e(vocs, X, Y):
            return np.random.rand(5, 2)

        def f(vocs, X, Y, x=None):
            return np.random.rand(5, 2)

        def g(v, X, Y):
            return np.random.rand(5, 2)

        def h(vocs, X, Y, Z):
            return np.random.rand(5, 2)

        def l(vocs, X, Y):
            return np.random.rand(5, 3)

        data = pd.DataFrame(np.random.rand(5, 3),
                            columns=list(self.vocs['variables']) +
                                    list(self.vocs['objectives']))

        for ele in [a, b, c, d, e, f]:
            alg = FunctionalGenerator(self.vocs, ele)
            alg.generate(data)

        for ele in [g, h, l]:
            with pytest.raises(BadFunctionError):
                alg = FunctionalGenerator(self.vocs, ele)
                alg.generate(data)
