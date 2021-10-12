import pandas as pd
from .routine import Routine
from .utils import transform_data
from ..algorithms.random import RandomSample


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
        samples = rs.generate(None)
        self.evaluator.submit_samples(samples)
        results = self.evaluator.collect_results()
        self.data = transform_data(results,
                                   self.vocs)

        # do optimization loop
        while not self.algorithm.is_terminated():
            samples = self.algorithm.generate(self.data)
            self.evaluator.submit_samples(samples)

            # gather results when all done
            results = self.evaluator.collect_results()

            # add transformed results to dataframe
            results = transform_data(results, self.vocs)

            # concatenate results
            self.data = pd.concat([self.data, results], ignore_index=True)

            # save results to file
            self.save_data()

        return self.data
