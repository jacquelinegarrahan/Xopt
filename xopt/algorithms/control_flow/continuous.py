import concurrent.futures
import logging
import queue

import pandas as pd

from xopt.algorithms.generators.generator import ContinuousGenerator
from xopt.algorithms.generators.random import RandomSample
from xopt.utils import BadDataError

logger = logging.getLogger(__name__)


def run_continuous(algorithm):
    """
    Run continuous (asynchronous) algorithm
    """
    # create queue
    q = queue.Queue()

    # add random initial samples to queue
    rs = RandomSample(algorithm.vocs, algorithm.n_initial_samples, 1)
    initial_samples = rs.generate(algorithm.data)
    for idx, sample in initial_samples.iterrows():
        q.put(sample)

    # run optimization
    while True:
        # if q is not empty submit samples to executor
        while not q.empty():
            sample = q.get()
            logger.debug(f"submitted sample {sample.values}")
            algorithm.evaluator.submit_samples(sample)

        # process done futures
        results, valid = algorithm.evaluator.collect_results(
            concurrent.futures.FIRST_COMPLETED
        )
        if results is not None:
            # add transformed results to dataframe
            results = algorithm.generator.transform_data(results)
            n_results = len(results)
            if not valid:
                logger.warning(
                    "No valid results from measurements, see save file for " "details"
                )

            # concatenate results
            if algorithm.data is not None:
                algorithm.data = pd.concat([algorithm.data, results], ignore_index=True)
            else:
                algorithm.data = results

            # save results to file
            algorithm.save_data()

            # add new items to the queue to replace old ones
            if not algorithm.generator.is_terminated():
                # if we can specify the number of samples then do so, otherwise
                # get one sample at a time to fill
                if issubclass(type(algorithm.generator), ContinuousGenerator):
                    algorithm.generator.set_n_samples(n_results)
                    new_samples = algorithm.generator.generate(algorithm.data)
                    for idx, sample in new_samples.iterrows():
                        q.put(sample)

                else:
                    for _ in range(n_results):
                        if not algorithm.generator.is_terminated():
                            new_sample = algorithm.generator.generate(algorithm.data)
                            if len(new_sample) != 1:
                                raise BadDataError(
                                    "generator must return only one " "sample"
                                )
                            q.put(new_sample)

        # end the optimization loop if we have run out of futures
        # (they are all done)
        if not algorithm.evaluator.futures and algorithm.generator.is_terminated():
            logger.info("Budget exceeded and simulations finished")
            break

    return algorithm.data
