import logging
import time

import pandas as pd

from xopt.generators.random import RandomSample

logger = logging.getLogger(__name__)


def run_batched(algorithm):
    """
    Run batched algorithm
    """
    # create initial samples
    rs = RandomSample(algorithm.vocs, algorithm.n_initial_samples)

    # generate a set of samples that has at least one valid sample
    logger.info('Generating and submitting initial samples')
    results = None
    while True and algorithm.n_initial_samples:
        samples = rs.generate(algorithm.data)
        algorithm.evaluator.submit_samples(samples)
        results, vaild_flag = algorithm.evaluator.collect_results()
        if vaild_flag:
            break
        else:
            logger.warning('random sampling did not return any valid data')

    if algorithm.n_initial_samples:
        results = algorithm.generator.transform_data(results)
        if algorithm.data is not None:
            algorithm.data = pd.concat([algorithm.data, results], ignore_index=True)
        else:
            algorithm.data = results

    # transform data for use in point generation
    algorithm.data = algorithm.generator.transform_data(algorithm.data)

    # do optimization loop
    while not algorithm.generator.is_terminated():
        logger.info('generating samples')
        t0 = time.time()

        samples = algorithm.generator.generate(algorithm.data)
        logger.debug(f'generated {len(samples)} samples in {time.time() - t0:.4} '
                     f'seconds')
        logger.debug(f'samples\n{samples}')
        algorithm.evaluator.submit_samples(samples)

        # gather results when all done
        logger.info('collecting results')
        results, valid = algorithm.evaluator.collect_results()

        if not valid:
            logger.warning('No valid results from measurements, see save file for '
                           'details')
        # concatenate results
        algorithm.data = pd.concat([algorithm.data, results], ignore_index=True)

        # transform data for use in point generation next loop
        algorithm.data = algorithm.generator.transform_data(algorithm.data)

        # save results to file
        algorithm.save_data()

    return algorithm.data
