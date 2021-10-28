import copy

import pytest

from xopt import Xopt
from xopt.generators.bayesian.utils import UnsupportedError
from xopt.tests.test_functions import TNK


class TestClassBayesExp:
    VOCS = TNK.VOCS
    config = {"vocs": TNK.VOCS.copy()}
    config["evaluate"] = {
        "name": "test_TNK",
        "function": "xopt.tests.test_functions.TNK.evaluate_TNK",
    }
    config["xopt"] = {"output_path": ""}
    config["algorithm"] = {
        "name": "bayesian_exploration",
        "options": {
            "n_initial_samples": 2,
            "n_steps": 2,
            "mc_samples": 4,
        },
    }

    def test_bayes_exp_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        X.run()

    def test_biasing(self):
        # run with proximal term
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)

        X.config["algorithm"]["options"]["sigma"] = [1, 1]
        X.run()

        # run with bad proximal term
        X.config["algorithm"]["options"]["sigma"] = [1, 1, 1]
        with pytest.raises(ValueError):
            X.run()

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config["algorithm"]["options"]["batch_size"] = 2

        X = Xopt(test_config)
        X.run()

        # try to add proximal term
        X.config["algorithm"]["options"]["sigma"] = [1, 1]
        with pytest.raises(ValueError):
            X.run()
