from copy import deepcopy

from ..xopt import Xopt
import pytest


class TestXoptConfig:
    YAML = """
    xopt:
      output_path: ''        

    algorithm:
      name: upper_confidence_bound
      options:  
        n_steps: 10
        n_initial_samples: 5

    evaluate: 
      name: quad_3d
      function: xopt.tests.test_functions.quad_3d.evaluate

    vocs:
      variables:
        x1: [0, 1]
        x2: [0, 1]
        x3: [0, 1]
      objectives: {y1: MINIMIZE}
      constraints: {}
      linked_variables: {}
      constants: {a: dummy_constant}

    """

    def test_read_config(self):
        # use hard coded YAML file above to test
        X = Xopt(self.YAML)
        X.configure_all()
        assert X.algorithm.n_initial_samples == 5
        assert X.vocs["variables"]["x3"] == [0, 1]

    def test_xopt_default(self):
        X = Xopt()

        # check default configs
        assert list(X.config.keys()) == ["xopt", "algorithm", "evaluate", "vocs"]

        # check bad config files
        bad_config = {"xopt": None, "generator": None}
        with pytest.raises(Exception):
            X = Xopt(bad_config)

    def test_bad_configs(self):
        X = Xopt()
        default_config = X.config

        # test allowable keys
        for name in default_config.keys():
            with pytest.raises(Exception):
                new_config = deepcopy(default_config)
                new_config[name].update({"random_key": None})
                X2 = Xopt(new_config)

    def test_algorithm_config(self):
        # test generator specification
        X = Xopt()
        with pytest.raises(ValueError):
            X.configure_algorithm()

        # retry with a bad function name
        with pytest.raises(Exception):
            X.algorithm_config["function"] = "dummy"
            X.configure_algorithm()

        # retry with bad module
        X.algorithm_config["function"] = "dummy.dummy"
        with pytest.raises(ModuleNotFoundError):
            X.configure_algorithm()

    def test_evaluate_config(self):
        X = Xopt()

        # default has no function
        with pytest.raises(ValueError):
            X.configure_evaluate()

        # specify a good function
        X.evaluate_config["function"] = lambda x: x
        X.configure_evaluate()

        # specify a bad function
        X.evaluate_config["function"] = lambda x, y: x
        with pytest.raises(ValueError):
            X.configure_evaluate()

        # specify a bad key
        X.evaluate_config["evaluate"] = None
        with pytest.raises(KeyError):
            X.configure_evaluate()
        del X.evaluate_config["evaluate"]

        def dummy(x, y=None):
            return x

        X.evaluate_config["function"] = dummy
        X.configure_evaluate()
        assert X.evaluate_config["options"] == {"y": None}

    def test_vocs_config(self):
        X = Xopt()
        X.configure_vocs()
