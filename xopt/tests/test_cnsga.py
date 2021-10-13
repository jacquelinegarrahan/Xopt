from xopt import Xopt
import yaml


class TestCNSGA:
    # Make a proper input file.
    YAML = """
    xopt: {output_path: null}

    generator:
      name: cnsga
      options: 
        max_generations: 50, 
        population_size: 128, 
        crossover_probability: 0.9, 
        mutation_probability: 1.0,
        selection: auto, 
        population: null

    simulation: 
      name: test_TNK
      evaluate: xopt.tests.test_functions.TNK.evaluate_TNK  

    vocs:
      name: TNK_test
      description: null
      simulation: test_TNK
      templates: null
      variables:
        x1: [0, 3.14159]
        x2: [0, 3.14159]
      objectives: 
        y1: MINIMIZE
        y2: MINIMIZE
      constraints:
        c1: [GREATER_THAN, 0]
        c2: [LESS_THAN, 0.5]
      linked_variables: {x9: x1}
      constants: {a: dummy_constant}

    """
    config = yaml.safe_load(YAML)

    def test_cnsga_tnk(self):
        X = Xopt(self.config)
        X.run()

    
    


