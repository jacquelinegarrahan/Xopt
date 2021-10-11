import pandas as pd
from .routine import Routine


class Batched(Routine):
    def __init__(self, config, evaluator, algorithm, output_path='.'):
        super(Batched, self).__init__(config, evaluator, algorithm, output_path)

    def run(self):
        """
        Run batched routine
        """

        while not self.algorithm.is_terminated():
            samples = self.algorithm.generate(self.data)
            self.evaluator.submit_samples(samples)

            # gather results when all done
            results = self.evaluator.collect_results()

            # concatenate results
            if self.data is not None:
                self.data = pd.concat([self.data, results], ignore_index=True)
            else:
                self.data = results

            # save results to file
            self.save_data()

        return self.data






