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


def get_mfkg(model,
             bounds,
             cost_aware_utility,
             optimize_options,
             n_variables,
             target_fidelities):

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
        num_restarts=optimize_options.get("num_restarts", 10),
        raw_samples=optimize_options.get("raw_samples", 1024),
        options=optimize_options.get("options", {"batch_limit": 10, "maxiter": 200}),
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


def create_mf_acq(model,
                  target_fidelities,
                  fixed_cost,
                  n_variables,
                  tkwargs,
                  base_acq,
                  optimize_options):
    """

    Create Multifidelity acquisition function

    """
    cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities,
                                         fixed_cost=fixed_cost)

    bounds = torch.zeros((2, n_variables), **tkwargs)
    bounds[1, :] = 1.0

    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    # get optimization options
    one_shot_options = copy.deepcopy(optimize_options)
    if "batch_initial_conditions" in one_shot_options:
        one_shot_options.pop("batch_initial_conditions")

    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=get_mfkg(model,
                              bounds,
                              cost_aware_utility,
                              base_acq,
                              optimize_options,
                              n_variables,
                              target_fidelities),
        bounds=bounds,
        **one_shot_options
    )

    optimize_options["batch_initial_conditions"] = X_init

    return get_mfkg(model,
                    bounds,
                    cost_aware_utility,
                    base_acq,
                    optimize_options,
                    n_variables,
                    target_fidelities)


def get_recommendation(model,
                       base_acq,
                       n_variables,
                       target_fidelities,
                       tkwargs,
                       optimize_options):
    bounds = torch.zeros((2, n_variables), **tkwargs)
    bounds[1, :] = 1.0
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=base_acq(model),
        d=n_variables,
        columns=list(target_fidelities.keys()),
        values=[1],
    )

    # get optimization options for recommendation optimization
    final_options = copy.deepcopy(optimize_options)
    for ele in ["q", "batch_initial_conditions"]:
        try:
            final_options.pop(ele)
        except KeyError:
            pass

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf, bounds=bounds[:, :-1], q=1, **final_options
    )

    return rec_acqf._construct_X_full(final_rec)
