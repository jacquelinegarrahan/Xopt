from typing import Dict, Union, Callable

from ..evaluators.evaluator import Evaluator
from ..generators.generator import Generator, ContinuousGenerator
from .algorithm import Algorithm


def run_algortihm(
    vocs: Dict,
    generator: Union[Generator, ContinuousGenerator],
    function: Callable = None,
    evaluator: Evaluator = None,
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
        raise ValueError("must either specify evaluator or callable function")

    # create algorithm arguments
    alg_config = {}
    alg_args = (vocs, evaluator, generator)

    # create and call algorithm
    alg = Algorithm(*alg_args, **kwargs)
    alg.run()

    return alg
