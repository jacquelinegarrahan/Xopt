import copy

import pytest
import torch

from xopt import Xopt
from xopt.tests.test_functions import TNK


class TestClassMOBO:
    VOCS = TNK.VOCS
    config = {"vocs": TNK.VOCS.copy()}
    config["evaluate"] = {
        "name": "test_TNK",
        "function": "xopt.tests.test_functions.TNK.evaluate_TNK",
    }
    config["xopt"] = {"output_path": ""}
    config["algorithm"] = {
        "name": "expected_hypervolume_improvement",
        "options": {
            "n_initial_samples": 2,
            "n_steps": 2,
            "mc_samples": 2,
            "ref": None,
        },
    }

    def test_mobo_base(self):
        test_config = copy.deepcopy(self.config)

        # try without reference point
        with pytest.raises(ValueError):
            X = Xopt(test_config)
            X.run()

        # try with bad reference point
        with pytest.raises(ValueError):
            test_config["algorithm"]["options"].update({"ref": {"y1": 1.4}})
            X = Xopt(test_config)
            X.run()

        # try with reference point
        test_config["algorithm"]["options"].update({"ref": {"y1": 1.4, "y2": 1.4}})
        X = Xopt(test_config)
        X.run()
        assert X.generator.ref == {"y1": -1.4, "y2": -1.4}

    def test_mobo_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config["algorithm"]["options"].update({"batch_size": 2})

        # try with reference point
        test_config["algorithm"]["options"].update({"ref": {"y1": 1.4, "y2": 1.4}})
        X = Xopt(test_config)

    def test_mobo_proximal(self):
        test_config = copy.deepcopy(self.config)
        test_config["algorithm"]["options"].update({"sigma": [1.0, 1.0]})
        test_config["algorithm"]["options"].update({"ref": {"y1": 1.4, "y2": 1.4}})

        # try with sigma matrix
        X = Xopt(test_config)
        X.run()

        # try with batched
        test_config["algorithm"]["options"].update({"batch_size": 2})
        with pytest.raises(ValueError):
            X = Xopt(test_config)

    def test_mobo_unconstrained(self):
        test_config = copy.deepcopy(self.config)
        test_config["vocs"]["constraints"] = {}
        test_config["algorithm"]["options"].update({"ref": {"y1": 1.4, "y2": 1.4}})

        # try with sigma matrix
        X = Xopt(test_config)
        X.run()
