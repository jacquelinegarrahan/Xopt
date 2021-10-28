import copy
import logging

import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import AffineFidelityCostModel
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf

# Logger
logger = logging.getLogger(__name__)


def get_mfkg(
    model,
    bounds,
    cost_aware_utility,
    num_restarts,
    optimize_options,
    n_variables,
    target_fidelities,
):

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=n_variables,
        columns=list(target_fidelities.keys()),
        values=[1],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=num_restarts,
        **optimize_options,
    )

    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=optimize_options.get("num_fantasies", 128),
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
        # X_pending=X_pending,
    )


def get_recommendation(
    model, n_variables, target_fidelities, tkwargs, num_restarts, optimize_options
):
    bounds = torch.zeros((2, n_variables), **tkwargs)
    bounds[1, :] = 1.0
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=n_variables,
        columns=list(target_fidelities.keys()),
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        num_restarts=num_restarts,
        q=1,
        **optimize_options,
    )

    return rec_acqf._construct_X_full(final_rec)
