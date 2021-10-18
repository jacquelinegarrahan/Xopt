import logging

import numpy as np
import copy
import torch
import pandas as pd
from botorch.acquisition import PosteriorMean, AcquisitionFunction, \
    InverseCostWeightedUtility
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import LinearMCObjective

from .base import BayesianGenerator
from .acquisition.mobo import get_corrected_ref, create_mobo_acqf
from .acquisition.exploration import create_bayes_exp_acq
from .acquisition.multi_fidelity import create_mf_acq, get_mfkg
from ...utils import check_dataframe
from ..utils import transform_data

logger = logging.getLogger(__name__)


class UpperConfidenceBound(BayesianGenerator):
    def __init__(self, vocs, n_steps=1, batch_size=1, beta=2.0, **kwargs):
        # need to specify a scalarized Objective to specify which index is the objective
        if len(vocs['objectives']) != 1:
            raise ValueError('cannot use UCB when multiple objectives are present')

        optimize_options = kwargs
        self.n_steps = n_steps
        super(UpperConfidenceBound, self).__init__(vocs,
                                                   qUpperConfidenceBound,
                                                   {},
                                                   optimize_options)

        weights = torch.zeros(len(vocs['variables']) + 1, **self.tkwargs)
        weights[-1] = 1.0
        sco = LinearMCObjective(weights)
        self.acquisition_function_options = {'beta': beta, 'objective': sco}

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

        self.acquisition_function_options = {'ref': ref_tensor,
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

        self.acquisition_function_options = {'n_constraints': len(vocs['constraints']),
                                             'n_variables': len(self.vocs['variables']),
                                             'sigma': sigma,
                                             'sampler': self.sampler,
                                             'q': batch_size}
        self._n_samples = batch_size

    def is_terminated(self):
        return self.n_calls >= self.n_steps


class MultiFidelity(BayesianGenerator):
    def __init__(self,
                 vocs,
                 budget=1,
                 batch_size=1,
                 fixed_cost=0.01,
                 **kwargs):

        # need to specify a scalarized Objective to specify which index is the objective
        if len(vocs['objectives']) != 1:
            raise ValueError('cannot use multi-fidelity BO when multiple objectives '
                             'are present')
        if 'cost' not in vocs['variables']:
            raise ValueError('multi-fidelity requires a `cost` variable in vocs')
        if vocs['variables']['cost'] != [0, 1]:
            raise RuntimeWarning('cost not normalized to [0, 1] range, proceed with '
                                 'caution')

        acq = create_mf_acq
        optimization_options = {'num_restarts': 20,
                                "raw_samples": 1024,
                                "num_fantasies": 128, }.update(kwargs)

        super(MultiFidelity, self).__init__(vocs, acq, {}, optimization_options)
        self.budget = budget

        # construct target fidelities dict
        tf = {}
        for idx, name in enumerate(vocs['variables']):
            if name == 'cost':
                tf[idx] = 1.0
        self.target_fidelities = tf
        self.fixed_cost = fixed_cost
        self.batch_size = batch_size

    def _get_and_optimize_acq(self, bounds) -> torch.Tensor:

        cost_model = AffineFidelityCostModel(fidelity_weights=self.target_fidelities,
                                             fixed_cost=self.fixed_cost)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=get_mfkg(self.model,
                                  bounds,
                                  cost_aware_utility,
                                  self.optimization_options,
                                  len(self.vocs['variables']),
                                  self.target_fidelities),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.optimization_options.get('num_restarts', 10),
            raw_samples=self.optimization_options.get('raw_samples', 512)
        )

        candidates, _ = optimize_acqf(
            acq_function=get_mfkg(self.model,
                                  bounds,
                                  cost_aware_utility,
                                  self.optimization_options,
                                  len(self.vocs['variables']),
                                  self.target_fidelities),
            bounds=bounds,
            q=self.batch_size,
            num_restarts=self.optimization_options.get('num_restarts', 10),
            raw_samples=self.optimization_options.get('raw_samples', 512),
            batch_initial_conditions=X_init,
            options={'batch_limit': 5, 'max_iter': 200}
        )
        return candidates

    def is_terminated(self):
        # calculate total cost and compare to budget
        if self._data is not None:
            return self._data['cost'].sum() > self.budget
        else:
            return False
