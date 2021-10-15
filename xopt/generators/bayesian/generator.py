import logging

import numpy as np
import torch
import pandas as pd
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler

from .base import BayesianGenerator
from .acquisition.mobo import get_corrected_ref, create_mobo_acqf
from .acquisition.exploration import create_bayes_exp_acq
from ...utils import check_dataframe
from ..utils import transform_data

logger = logging.getLogger(__name__)


class UpperConfidenceBound(BayesianGenerator):
    def __init__(self, vocs, n_steps=1, batch_size=1, beta=2.0, **kwargs):
        acquisition_options = {'beta': beta}
        optimize_options = kwargs
        self.n_steps = n_steps
        super(UpperConfidenceBound, self).__init__(vocs,
                                                   qUpperConfidenceBound,
                                                   acquisition_options,
                                                   optimize_options)
        self._n_samples = batch_size

    def is_terminated(self):
        return self.n_calls >= self.n_steps


class ExpectedHypervolumeImprovement(BayesianGenerator):
    def __init__(self, vocs, ref=None, n_steps=1, batch_size=1, sigma=None,
                 mc_samples=1024, **kwargs):
        acq = create_mobo_acqf
        optimize_options = kwargs
        optimize_options.update({'options':
                                     {"batch_limit": 5, "maxiter": 200,
                                      "nonnegative": True},
                                 'sequential': True, })
        super(ExpectedHypervolumeImprovement, self).__init__(vocs, acq,
                                                             {}, optimize_options)
        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)
        self.n_steps = n_steps
        if ref is not None:
            if list(ref) == list(self.vocs['objectives']):
                ref = get_corrected_ref(self.vocs, ref)
            else:
                raise ValueError('reference point does not correctly correspond to '
                                 'objectives in vocs')
        else:
            raise ValueError('need to specify a reference point')
        self.ref = ref
        ref_tensor = torch.tensor([self.ref[key] for key in self.ref], **self.tkwargs)

        if batch_size != 1 and sigma is not None:
            raise ValueError('cannot use multi-batch with proximal biasing')

        if sigma is not None:
            sigma = torch.tensor(sigma, **self.tkwargs)

        self.acqisition_function_options = {'ref': ref_tensor,
                                            'n_objectives': len(vocs['objectives']),
                                            'n_constraints': len(vocs['constraints']),
                                            'sigma': sigma,
                                            'sampler': self.sampler}
        self._n_samples = batch_size

    def is_terminated(self):
        return self.n_calls >= self.n_steps

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        overwrite how training data is transformed for MOBO, we want to normalize
        objective data with respect to the reference point
        """
        check_dataframe(data, self.vocs)
        new_df = transform_data(data, self.vocs)

        # overwrite objective transformations to normalize w.r.t. ref point
        for key in new_df.keys():
            if key in self.vocs['objectives']:
                if self.vocs['objectives'][key] == 'MINIMIZE':
                    new_df[key + '_t'] = -new_df[key] / self.ref[key]
                else:
                    new_df[key + '_t'] = new_df[key] / self.ref[key]

        return new_df


class BayesianExploration(BayesianGenerator):
    def __init__(self, vocs, n_steps=1, batch_size=1, sigma=None,
                 mc_samples=1024, **kwargs):
        acq = create_bayes_exp_acq
        optimize_options = kwargs
        optimize_options.update({'options':
                                     {"batch_limit": 5, "maxiter": 200,
                                      "nonnegative": True},
                                 'sequential': True, })

        super(BayesianExploration, self).__init__(vocs, acq, {}, optimize_options)

        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)
        self.n_steps = n_steps

        if batch_size != 1 and sigma is not None:
            raise ValueError('cannot use multi-batch with proximal biasing')

        if sigma is not None:
            sigma = torch.tensor(sigma, **self.tkwargs)
        else:
            sigma = torch.ones(len(self.vocs['variables']), **self.tkwargs) * 1e10

        self.acqisition_function_options = {'n_constraints': len(vocs['constraints']),
                                            'n_variables': len(self.vocs['variables']),
                                            'sigma': sigma,
                                            'sampler': self.sampler,
                                            'q': batch_size}
        self._n_samples = batch_size

    def is_terminated(self):
        return self.n_calls >= self.n_steps
