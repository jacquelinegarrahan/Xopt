from typing import Dict, Union, Callable
from ..generators.generator import Generator, ContinuousGenerator
from ..evaluators.evaluator import Evaluator
from .continuous import Continuous
from .batched import Batched


def run_algortihm(
        vocs: Dict,
        generator: Union[Generator, ContinuousGenerator],
        function: Callable = None,
        evaluator: Evaluator = None,
        algorithm_type: str = 'batched',
        **kwargs,
):
    """
    run algorithm in functional mode, for use in scripting in python
    creates an algorithm object and runs it, returning the algorithm object after
    finishing
    """

    # fill in default objects
    if evaluator is None and function is not None:
        evaluator = Evaluator(vocs, function)
    elif evaluator is not None and function is None:
        pass
    else:
        raise ValueError('must either specify evaluator or callable function')

    # create algorithm arguments
    alg_config = {}
    alg_args = (alg_config, vocs, evaluator, generator)

    # create and call algorithm
    alg_dict = {'batched': Batched, 'continuous': Continuous}
    if algorithm_type not in alg_dict:
        raise KeyError('specified algorithm type is not available')

    alg = alg_dict[algorithm_type](*alg_args, **kwargs)
    alg.run()

    return alg


