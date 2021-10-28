from xopt import Xopt

from botorch.acquisition.analytic import UpperConfidenceBound


class TestBayesianOptimization:
    # Make a proper input file.
    YAML = """
    xopt: 
        output_path: null

    algorithm:
      name: bayesian_optimization
      options:  
          n_initial_samples: 15
          n_steps: 2
          acquisition_function: xopt.tests.bayesian.custom_acq.acq

    simulation: 
      name: test
      evaluate: xopt.tests.test_functions.quad_3d.evaluate

    vocs:
      name: test
      variables:
        x1: [0, 1.0]
        x2: [0, 1.0]
        x3: [0, 1.0]
      objectives:
        y1: 'MINIMIZE'
      linked_variables: {}
      constants: {a: dummy_constant}

    """
    config = YAML

    def test_bayesian_optimization(self):
        # test generalized bayesian optimization from external acquisition function
        X = Xopt(self.config)
        X.run()
