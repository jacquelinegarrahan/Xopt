import yaml
from xopt import Xopt


class TestUCB:
    YAML = """
    xopt:
      output_path: ''

    algorithm:
      name: upper_confidence_bound
      options:  
        n_steps: 2
        n_initial_samples: 50

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
    config = yaml.safe_load(YAML)

    def test_ucb(self):
        # run a ucb optimization on the quad_3d test function
        X = Xopt(self.config)
        print(X.run())
