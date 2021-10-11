import logging
from abc import ABC, abstractmethod
from inspect import signature
from typing import Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class BadFunctionError(ValueError):
    pass


class Algorithm(ABC):
    def __init__(self, vocs: Dict):
        self.vocs = vocs

    @abstractmethod
    def generate(self, data) -> pd.DataFrame:
        """
        Generate points for observation based on algorithm state

        """
        pass

    @abstractmethod
    def is_terminated(self):
        """
        return True if algorithm should terminate
        """
        return False


class FunctionalAlgorithm(Algorithm):
    """
    Algorithm class that takes in an arbitrary function where we will
    check if the function takes one of the following forms:
    - f(vocs: Dict, **kwargs) -> np.ndarray
    - f(vocs: Dict, data: pandas.Dataframe, **kwargs) -> np.ndarray
    - f(vocs: Dict, X: np.ndarray, Y: np.ndarray, **kwargs) -> np.ndarray
    NOTE: typing is NOT enforced
    and outputs points have the correct shapes with respect to vocs. Algorithm
    terminates when the number of max calls is exceeded.
    """

    def __init__(self,
                 vocs: Dict,
                 f: Callable,
                 options: Dict = None
                 ):
        self.f = f
        self.max_calls = options.pop('max_calls', 1)
        self.n_calls = 0
        self.options = options or {}
        super(FunctionalAlgorithm, self).__init__(vocs)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:

        # check signature of callable to pass callable correct data
        sig = signature(self.f)
        sig_pos_parameters = []
        results = None
        for name, ele in sig.parameters.items():
            if ele.kind == ele.POSITIONAL_OR_KEYWORD and ele.default is ele.empty:
                sig_pos_parameters += [name]

        if sig_pos_parameters == ['vocs']:
            results = self.f(self.vocs, **self.options)
        elif sig_pos_parameters == ['vocs', 'data']:
            results = self.f(self.vocs, data, **self.options)
        elif sig_pos_parameters == ['vocs', 'X', 'Y']:
            # convert pandas dataframe to numpy
            X = data[self.vocs['variables']].to_numpy()
            Y = data[self.vocs['objectives']].to_numpy()

            if 'constraints' in self.vocs:
                C = data[self.vocs['constraints']].to_numpy()
                try:
                    results = self.f(self.vocs, X, Y, C=C, **self.options)
                except ValueError:
                    logger.error('callable function does not support constraints with '
                                 'keyword `C`')

            else:
                results = self.f(self.vocs, X, Y, **self.options)
        else:
            raise BadFunctionError('callable function input arguments not correct, '
                                   'must be one of the following forms:'
                                   '- f(vocs: Dict, **kwargs) -> np.ndarray\n'
                                   '- f(vocs: Dict, data: pandas.Dataframe, **kwargs) '
                                   '-> np.ndarray\n'
                                   '- f(vocs: Dict, X: np.ndarray, Y: np.ndarray, '
                                   '**kwargs) -> np.ndarray\n'
                                   'NOTE: typing is NOT enforced')

        if results.shape[-1] != len(self.vocs['variables']):
            raise BadFunctionError('callable function does not return the correct '
                                   f'dimensional array, returned {results.shape} but '
                                   f'needs to match # of variables in vocs')

        self.n_calls += 1
        return pd.DataFrame(results, columns=self.vocs['variables'])

    def is_terminated(self):
        return self.n_calls >= self.max_calls
