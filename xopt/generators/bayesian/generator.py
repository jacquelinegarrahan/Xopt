import logging

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from .base import BayesianGenerator
from .acquisition.mobo import get_corrected_ref, create_mobo_acqf

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
    def __init__(self, vocs, ref=None, n_steps=1, batch_size=1, sigma=None, **kwargs):
        acq = create_mobo_acqf
        optimize_options = kwargs
        super(ExpectedHypervolumeImprovement, self).__init__(vocs, acq,
                                                             {}, optimize_options)
        self.n_steps = n_steps
        if ref is not None:
            if len(ref) == len(self.vocs['objectives']):
                ref = get_corrected_ref(self.vocs, torch.tensor(ref, **self.tkwargs))
            else:
                raise ValueError('reference point incorrect shape for vocs')
        else:
            raise ValueError('need to specify a reference point')

        if batch_size != 1 and sigma is not None:
            raise ValueError('cannot use multi-batch with proximal biasing')

        if sigma is not None:
            sigma = torch.tensor(sigma, **self.tkwargs)

        self.acqisition_function_options = {'ref': ref,
                                            'n_objectives': len(vocs['objectives']),
                                            'n_constraints': len(vocs['constraints']),
                                            'sigma': sigma}
        self._n_samples = batch_size

    def is_terminated(self):
        return self.n_calls >= self.n_steps
