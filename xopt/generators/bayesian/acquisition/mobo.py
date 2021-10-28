import logging
from functools import partial

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from ..acquisition.proximal import ProximalAcquisitionFunction

# Logger

logger = logging.getLogger(__name__)


def get_corrected_ref(vocs, ref):
    for key in vocs["objectives"]:
        if vocs["objectives"][key] == "MINIMIZE":
            ref[key] = -ref[key]
    return ref


def create_mobo_acqf(
    model: Model,
    ref: torch.Tensor,
    n_objectives: int,
    n_constraints: int,
    sigma: torch.Tensor = None,
    sampler=None,
) -> AcquisitionFunction:
    train_outputs = model.train_targets.T
    train_y = train_outputs[:, :n_objectives]
    train_c = train_outputs[:, n_objectives:]

    # compute feasible observations
    is_feas = (train_c <= 0).all(dim=-1)

    # compute points that are better than the known reference point
    better_than_ref = (train_y > ref).all(dim=-1)
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(
        ref_point=ref,
        # use observations that are better than the specified reference point and
        # feasible
        Y=train_y[better_than_ref & is_feas],
    )

    # define constraint functions - note issues with lambda implementation
    # https://tinyurl.com/j8wmckd3
    def constr_func(Z, index=-1):
        return Z[..., index]

    constraint_functions = []
    for i in range(1, n_constraints + 1):
        constraint_functions += [partial(constr_func, index=-i)]

    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref.tolist(),  # use known reference point
        partitioning=partitioning,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
        constraints=constraint_functions,
        sampler=sampler,
    )

    # add in proximal biasing
    if sigma is not None:
        acq_func = ProximalAcquisitionFunction(acq_func, sigma)

    return acq_func
