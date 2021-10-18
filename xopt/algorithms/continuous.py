import logging
import concurrent.futures
from typing import Dict, Union
import pandas as pd
import queue
from .algorithm import Algorithm
from ..evaluators.evaluator import Evaluator
from ..utils import BadDataError
from ..generators.generator import Generator
from ..generators.random import RandomSample
from ..generators.generator import ContinuousGenerator

logger = logging.getLogger(__name__)


class Continuous(Algorithm):
    def __init__(self,
                 config: Dict,
                 vocs: Dict,
                 evaluator: Evaluator,
                 generator: Union[Generator, ContinuousGenerator],
                 n_processes=1,
                 output_path='.'):
        self.n_inital_samples = n_processes
        super(Continuous, self).__init__(config, vocs, evaluator, generator,
                                         output_path)

    def run(self):
        """
        Run continuous (asynchronous) algorithm
        """
        # create queue
        q = queue.Queue()

        #add random initial samples to queue
        rs = RandomSample(self.vocs, self.n_inital_samples, 1)
        initial_samples = rs.generate(self.data)
        for idx, sample in initial_samples.iterrows():
            q.put(sample)

        # run optimization
        while True:
            # if q is not empty submit samples to executor
            while not q.empty():
                sample = q.get()
                logger.debug(f'submitted sample {sample.values}')
                self.evaluator.submit_samples(sample)

            # process done futures
            results, valid = self.evaluator.collect_results(
                concurrent.futures.FIRST_COMPLETED)
            if results is not None:
                # add transformed results to dataframe
                results = self.generator.transform_data(results)
                n_results = len(results)
                if not valid:
                    logger.warning(
                        'No valid results from measurements, see save file for '
                        'details')

                # concatenate results
                if self.data is not None:
                    self.data = pd.concat([self.data, results], ignore_index=True)
                else:
                    self.data = results

                # save results to file
                self.save_data()

                # add new items to the queue to replace old ones
                if not self.generator.is_terminated():
                    # if we can specify the number of samples then do so, otherwise
                    # get one sample at a time to fill
                    if issubclass(type(self.generator), ContinuousGenerator):
                        self.generator.set_n_samples(n_results)
                        new_samples = self.generator.generate(self.data)
                        for idx, sample in new_samples.iterrows():
                            q.put(sample)

                    else:
                        for _ in range(n_results):
                            if not self.generator.is_terminated():
                                new_sample = self.generator.generate(self.data)
                                if len(new_sample) != 1:
                                    raise BadDataError('generator must return only one '
                                                       'sample')
                                q.put(new_sample)

            # end the optimization loop if we have run out of futures
            # (they are all done)
            if not self.evaluator.futures and self.generator.is_terminated():
                logger.info("Budget exceeded and simulations finished")
                break

        return self.data






