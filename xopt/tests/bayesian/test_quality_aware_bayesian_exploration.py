from xopt import Xopt


class TestQualityAware:
    YAML = """
    xopt: {output_path: null}

    algorithm:
      name: quality_aware_exploration
      options:  
          n_initial_samples: 50
          n_steps: 1
          target_observation: y1
          quality_observation: q1
          nominal_quality_parameters: 
            x3
            x4

    evaluate:
      name: test_TNK
      function: xopt.tests.test_functions.quality_aware.evaluate_4d

    vocs:
      name: TNK_test
      variables:
        x1: [0, 1.0]
        x2: [0, 1.0]
        x3: [0, 1.0]
        x4: [0, 1.0]
      objectives:
        y1: None
        q1: MAXIMIZE

      constraints: {}
      linked_variables: {}
      constants: {}

    """
    config = YAML

    def test_quality_aware(self):
        # test generalized bayesian optimization from external acquisition function
        X = Xopt(self.config)
        results = X.run()
