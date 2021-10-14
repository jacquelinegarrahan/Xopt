import logging

import pandas as pd

from .algorithm import Algorithm
from ..utils import BadDataError
from ..generators.random import RandomSample

logger = logging.getLogger(__name__)


class Batched(Algorithm):
    def __init__(self, config, evaluator, generator,
                 n_initial_samples=1, output_path='.'):
        self.n_inital_samples = n_initial_samples
        super(Batched, self).__init__(config, evaluator, generator, output_path)

    def run(self):
        """
        Run batched algorithm
        """
        # create initial samples
        rs = RandomSample(self.vocs, self.n_inital_samples)

        # generate a set of samples that has at least one valid sample
        while True:
            samples = rs.generate(self.data)
            self.evaluator.submit_samples(samples)
            results, vaild_flag = self.evaluator.collect_results()
            if vaild_flag:
                break
            else:
                logger.warning('random sampling did not return any valid data')

        self.data = self.generator.transform_data(results)

        # do optimization loop
        while not self.generator.is_terminated():
            samples = self.generator.generate(self.data)
            self.evaluator.submit_samples(samples)

            # gather results when all done
            results, valid = self.evaluator.collect_results()
            # add transformed results to dataframe
            results = self.generator.transform_data(results)
            if not valid:
                logger.warning('No valid results from measurements, see save file for '
                               'details')
            # concatenate results
            self.data = pd.concat([self.data, results], ignore_index=True)

            # save results to file
            self.save_data()

        return self.data
