import logging

import pandas as pd

from .routine import Routine
from ..utils import BadDataError
from ..algorithms.random import RandomSample

logger = logging.getLogger(__name__)


class Batched(Routine):
    def __init__(self, config, evaluator, algorithm,
                 n_initial_samples=1, output_path='.'):
        self.n_inital_samples = n_initial_samples
        super(Batched, self).__init__(config, evaluator, algorithm, output_path)

    def run(self):
        """
        Run batched routine
        """
        # create initial samples
        rs = RandomSample(self.vocs, self.n_inital_samples)

        # generate a set of samples that has at least one valid sample
        while True:
            try:
                samples = rs.generate(self.data)
                self.evaluator.submit_samples(samples)
                results = self.evaluator.collect_results()
                break
            except BadDataError:
                logger.warning('random sampling did not return any valid data')

        self.data = self.algorithm.transform_data(results)

        # do optimization loop
        while not self.algorithm.is_terminated():
            samples = self.algorithm.generate(self.data)
            self.evaluator.submit_samples(samples)

            try:
                # gather results when all done
                results = self.evaluator.collect_results()

                # add transformed results to dataframe
                results = self.algorithm.transform_data(results)

                # concatenate results
                self.data = pd.concat([self.data, results], ignore_index=True)

            except BadDataError:
                samples['status'] = 'error'
                self.data = pd.concat([self.data, samples], ignore_index=True)
                logger.warning('No valid results from measurements, see save file for '
                               'details')

            # save results to file
            self.save_data()

        return self.data
