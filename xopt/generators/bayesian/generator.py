import logging
from copy import deepcopy

import pandas as pd
import torch
from botorch.acquisition import InverseCostWeightedUtility
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import LinearMCObjective
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.sampling import SobolQMCNormalSampler

from .acquisition.exploration import create_bayes_exp_acq
from .acquisition.mobo import get_corrected_ref, create_mobo_acqf
from .acquisition.multi_fidelity import get_mfkg, get_recommendation
from .models.models import create_multi_fidelity_model
from .base import BayesianGenerator
from ..utils import transform_data, untransform_x
from ...utils import check_dataframe

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
                 num_restarts: int = 20,
                 raw_samples: int = 1024,
                 num_fantasies: int = 128,
                 **kwargs) -> None:

        # need to specify a scalarized Objective to specify which index is the objective
        if len(vocs['objectives']) != 1:
            raise ValueError('cannot use multi-fidelity BO when multiple objectives '
                             'are present')
        if 'cost' not in vocs['variables']:
            raise ValueError('multi-fidelity requires a `cost` variable in vocs')
        if vocs['variables']['cost'] != [0, 1]:
            raise RuntimeWarning('cost not normalized to [0, 1] range, proceed with '
                                 'caution')

        optimization_options = {"raw_samples": raw_samples,
                                'options': {"batch_limit": 10, "maxiter": 200}
                                }

        super(MultiFidelity, self).__init__(vocs,
                                            None,
                                            {},
                                            optimization_options,
                                            create_model_f=create_multi_fidelity_model)
        self.budget = budget

        # construct target fidelities dict
        tf = {}
        for idx, name in enumerate(vocs['variables']):
            if name == 'cost':
                tf[idx] = 1.0
        self.target_fidelities = tf
        self.fixed_cost = fixed_cost
        self.set_n_samples(batch_size)

        self.num_restarts = num_restarts
        self.num_fantasies = num_fantasies

    def _get_and_optimize_acq(self, bounds) -> torch.Tensor:

        cost_model = AffineFidelityCostModel(fidelity_weights=self.target_fidelities,
                                             fixed_cost=self.fixed_cost)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        mfkg_acqf = get_mfkg(self.model,
                             bounds,
                             cost_aware_utility,
                             self.num_restarts,
                             self.optimization_options,
                             len(self.vocs['variables']),
                             self.target_fidelities)

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=mfkg_acqf,
            bounds=bounds,
            q=self._n_samples,
            num_restarts=self.num_restarts,
            raw_samples=self.optimization_options['raw_samples']
        )

        oo_copy = deepcopy(self.optimization_options)
        oo_copy.update({'batch_initial_conditions': X_init})
        candidates, _ = optimize_acqf(
            acq_function=mfkg_acqf,
            bounds=bounds,
            q=self._n_samples,
            num_restarts=self.num_restarts,
            **oo_copy
        )
        return candidates

    def get_recommendation(self, data):
        model = self._create_model(data)
        result = get_recommendation(model,
                                    len(self.vocs['variables']),
                                    self.target_fidelities,
                                    self.tkwargs,
                                    self.num_restarts,
                                    self.optimization_options)

        result = result.detach().cpu().numpy()
        return untransform_x(self.numpy_to_dataframe(result), self.vocs)

    def is_terminated(self):
        # calculate total cost and compare to budget
        if self._data is not None:
            return (self._data['cost'] + self.fixed_cost).sum() > self.budget
        else:
            return False
