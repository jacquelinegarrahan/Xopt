import copy

from xopt import Xopt
from .test_functions.multi_fidelity import VOCS


class TestClassMultiFidelity:
    config = {'vocs': VOCS.copy(),
              'evaluate': {'name': 'AugmentedHartmann',
                           'function':
                               'xopt.tests.test_functions.multi_fidelity.evaluate'},
              'xopt': {'output_path': ''},
              'algorithm': {'name': 'multi_fidelity',
                            'options': {
                                'budget': 2,
                                'fixed_cost': 1.0,
                                'num_restarts': 2,
                                "raw_samples": 4,
                                "num_fantasies": 4,
                            }}}

    def test_multi_fidelity_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        X.run()

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['algorithm']['type'] = 'continuous'
        test_config['algorithm']['options']['n_processes'] = 2
        test_config['algorithm']['options']['budget'] = 4

        X = Xopt(test_config)
        print(X.run())
