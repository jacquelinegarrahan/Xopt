from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable, Dict
import inspect
from inspect import signature, Parameter

import logging

logger = logging.getLogger(__name__)


class BadFunctionError(ValueError):
    pass


class Algorithm(ABC):
    def __init__(self, vocs: Dict):
        self.vocs = vocs

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate points for observation based on algorithm state

        """
        pass


class FunctionalAlgorithm(Algorithm):
    """
    Algorithm class that takes in an arbitrary function where we will
    check if the function takes one of the following forms:
    - f(vocs: Dict, **kwargs) -> np.ndarray
    - f(vocs: Dict, data: pandas.Dataframe, **kwargs) -> np.ndarray
    - f(vocs: Dict, X: np.ndarray, Y: np.ndarray, **kwargs) -> np.ndarray
    NOTE: typing is NOT enforced
    and outputs points have the correct shapes with respect to vocs
    """

    def __init__(self,
                 vocs: Dict,
                 f: Callable,
                 ):
        self.f = f
        super(FunctionalAlgorithm, self).__init__(vocs)

    def generate(self,
                 data: pd.DataFrame,
                 **kwargs) -> pd.DataFrame:

        # check signature of callable to pass callable correct data
        sig = signature(self.f)
        sig_pos_parameters = []
        for name, ele in sig.parameters.items():
            if ele.kind == ele.POSITIONAL_OR_KEYWORD and ele.default is ele.empty:
                sig_pos_parameters += [name]

        if sig_pos_parameters == ['vocs']:
            results = self.f(self.vocs, **kwargs)
        elif sig_pos_parameters == ['vocs', 'data']:
            results = self.f(self.vocs, data, **kwargs)
        elif sig_pos_parameters == ['vocs', 'X', 'Y']:
            # convert pandas dataframe to numpy
            X = data[self.vocs['variables']].to_numpy()
            Y = data[self.vocs['objectives']].to_numpy()

            if 'constraints' in self.vocs:
                C = data[self.vocs['constraints']].to_numpy()
                try:
                    results = self.f(self.vocs, X, Y, C=C, **kwargs)
                except ValueError:
                    logger.error('callable function does not support constraints with '
                                 'keyword `C`')

            else:
                results = self.f(self.vocs, X, Y, **kwargs)
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

        return pd.DataFrame(results, columns=self.vocs['variables'])
