import copy

from xopt import Xopt
from .test_functions.multi_fidelity import VOCS, evaluate


class TestClassMultiFidelity:
    config = {'vocs': VOCS.copy()}
    config['evaluate'] = {'name': 'AugmentedHartmann',
                            'function':
                                'xopt.tests.test_functions.multi_fidelity.evaluate'}
    config['xopt'] = {'output_path': ''}
    config['algorithm'] = {'name': 'multi_fidelity',
                           'options': {
                               'budget': 2,
                           }}

    def test_multi_fidelity_base(self):
        test_config = copy.deepcopy(self.config)
        X = Xopt(test_config)
        print(X.run())

    def test_batch(self):
        test_config = copy.deepcopy(self.config)
        test_config['generator']['options']['processes'] = 2

        X = Xopt(test_config)
        X.run()
