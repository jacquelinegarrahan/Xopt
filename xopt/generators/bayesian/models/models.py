import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from ..models.nan_enabled import get_nan_model
from ..outcome_transforms import NanEnabledStandardize
from xopt.vocs_tools import get_bounds


def create_model(train_data, vocs, custom_model=None, **kwargs):
    train_x = train_data['X']
    train_y = train_data['Y']
    train_c = train_data.get('C', None)

    input_normalize = Normalize(
        len(vocs["variables"]),
        torch.tensor(
            get_bounds(vocs),
        ).to(train_x),
    )

    # create model
    if custom_model is None:
        if train_c is not None:
            train_outputs = torch.hstack((train_y, train_c))
        else:
            train_outputs = train_y

        # test if there are nans in the training data
        if torch.any(torch.isnan(train_outputs)):
            output_standardize = NanEnabledStandardize(m=1)
            model = get_nan_model(
                train_x, train_outputs, input_normalize, output_standardize
            )
        else:
            output_standardize = None#Standardize(m=train_outputs.shape[-1])
            model = SingleTaskGP(
                train_x,
                train_outputs,
                input_transform=None,
                outcome_transform=output_standardize,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

    else:
        model = custom_model(train_x, train_y, train_c, vocs, **kwargs)
        assert isinstance(model, Model)

    return model


def create_multi_fidelity_model(train_data, vocs):
    train_x = train_data['X']
    train_y = train_data['Y']
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        input_transform=None,
        outcome_transform=None,
        data_fidelity=list(vocs['variables'].keys()).index('cost'),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
