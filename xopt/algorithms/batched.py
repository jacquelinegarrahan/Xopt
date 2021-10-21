import logging
import time

import pandas as pd

from .algorithm import Algorithm
from ..generators.random import RandomSample

logger = logging.getLogger(__name__)


class Batched(Algorithm):
    def __init__(self, config, vocs, evaluator, generator,
                 n_initial_samples=1, output_path='.', restart_file=None):
        self.n_initial_samples = n_initial_samples
        super(Batched, self).__init__(config, vocs, evaluator, generator,
                                      output_path, restart_file)

    def run(self):
        """
        Run batched algorithm
        """
        # create initial samples
        rs = RandomSample(self.vocs, self.n_initial_samples)

        # generate a set of samples that has at least one valid sample
        logger.info('Generating and submitting initial samples')
        results = None
        while True and self.n_initial_samples:
            samples = rs.generate(self.data)
            self.evaluator.submit_samples(samples)
            results, vaild_flag = self.evaluator.collect_results()
            if vaild_flag:
                break
            else:
                logger.warning('random sampling did not return any valid data')

        if self.n_initial_samples:
            results = self.generator.transform_data(results)
            if self.data is not None:
                self.data = pd.concat([self.data, results], ignore_index=True)
            else:
                self.data = results

        # do optimization loop
        while not self.generator.is_terminated():
            logger.info('generating samples')
            t0 = time.time()
            samples = self.generator.generate(self.data)
            logger.debug(f'generated {len(samples)} samples in {time.time() - t0:.4} '
                         f'seconds')
            logger.debug(f'samples\n{samples}')
            self.evaluator.submit_samples(samples)

            # gather results when all done
            logger.info('collecting results')
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
