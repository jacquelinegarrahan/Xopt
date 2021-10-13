import concurrent.futures
from typing import Dict, Callable
from concurrent.futures import Executor
import traceback
from ..utils import check_dataframe
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def sampler_evaluate(inputs: Dict,
                     evaluate_f: Callable,
                     **eval_args):
    """
    Wrapper to catch any exceptions

    Parameters
    ----------
    inputs: Dict
        possible inputs to evaluate_f (a single positional argument)
    evaluate_f: Callable
        a function that takes a dict with keys, and returns some output

    """
    outputs = None

    try:
        outputs = evaluate_f(inputs, **eval_args)

    except Exception as ex:
        # No need to print a nasty exception
        logger.error(f"Exception caught in {__name__}")
        outputs = {
            "Exception": str(ex),
            #"Traceback": traceback.format_exc(ex),
        }

    finally:
        result = {**inputs, **outputs}

    return result


class Evaluator:
    def __init__(self,
                 vocs: Dict,
                 f: Callable,
                 executor: Executor,
                 evaluate_options: Dict = None
                 ):
        self.f = f
        self.vocs = vocs
        self.evaluate_options = evaluate_options or {}
        self.executor = executor
        self.futures = {}

    def submit_samples(self, samples):
        """
        Submit samples to executor

        Parameters
        ----------
        samples : np.ndarray
            Samples to be evaluated, should be 2D and the last axis should have the
            same length of len(vocs['variables'])
        """
        self.futures = {}

        assert len(samples.shape) == 2
        assert samples.shape[-1] == len(self.vocs['variables'])

        for idx, sample in samples.iterrows():
            setting = sample.copy()
            if self.vocs['constants']:
                setting.update(self.vocs['constants'])

            self.futures.update(
                {
                    self.executor.submit(
                        sampler_evaluate, setting, self.f, **self.evaluate_options
                    ): sample
                }
            )

    def collect_results(self,
                        return_when=concurrent.futures.ALL_COMPLETED):
        """
        Get all of the results from the futures
        """

        done, not_done = concurrent.futures.wait(
            self.futures, timeout=0.1, return_when=return_when
        )

        # from the finished futures collect the data into a pandas dataframe
        results = []
        for future in done:
            r = future.result()
            if 'Exception' in r:
                r.update({'status': 'error'})
            else:
                r.update({'status': 'done'})
            results += [r]

        # create dataframe
        data = pd.DataFrame(results)

        # check to make sure at least one valid measurement has been made, else raise
        # an exception
        check_dataframe(data, self.vocs)

        return data



