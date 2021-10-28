import logging
from abc import ABC
from typing import Dict, Callable, Union

import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.optim.optimize import optimize_acqf

from .models.models import create_model
from ..generator import ContinuousGenerator
from ..utils import untransform_x
from ...tools import get_function_defaults, get_function
from ...utils import check_and_fill_defaults

logger = logging.getLogger(__name__)


class BayesianGenerator(ContinuousGenerator):
    def __init__(
        self,
        vocs: Dict,
        acquisition_function: Union[Callable, str] = None,
        acquisition_options: Dict = None,
        optimize_options: Dict = None,
        create_model_f: Callable = create_model,
        n_steps: int = 1,
    ):
        """
        General Bayesian optimization generator
        NOTE: we assume maximization for all acquisition functions
        """
        super(BayesianGenerator, self).__init__(vocs)
        self.model = None

        # if acquisition function is a string try to get the callable using function
        # import
        if isinstance(acquisition_function, str):
            self.acquisition_function = get_function(acquisition_function)
        else:
            self.acquisition_function = acquisition_function

        self.acquisition_function_options = acquisition_options or {}
        self.optimization_options = optimize_options or {}
        self.create_model_f = create_model_f
        self.n_steps = n_steps

        # get optimization kwargs defaults
        optimization_defaults = get_function_defaults(optimize_acqf)

        self.optimization_options = check_and_fill_defaults(
            self.optimization_options, optimization_defaults
        )

        self.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

        # set up gpu if requested
        use_gpu = self.acquisition_function_options.get("use_gpu", False)
        if use_gpu:
            if torch.cuda.is_available():
                self.tkwargs["device"] = torch.device("cuda")
                logger.info(
                    f"using gpu device "
                    f'{torch.cuda.get_device_name(self.tkwargs["device"])}'
                )
            else:
                logger.warning("gpu requested but not found, using cpu")

        # optimize the acquisition function in normalized space
        self.bounds = torch.zeros(2, len(self.vocs["variables"]), **self.tkwargs)
        self.bounds[1, :] = 1.0

        self._data = None

    def is_terminated(self):
        return self.n_calls >= self.n_steps

    def get_acqf(self, model, **kwargs):
        acq_func = self.acquisition_function(model, **kwargs)

        if not isinstance(acq_func, AcquisitionFunction):
            raise RuntimeError(
                "callable `acquisition_function` does not return type "
                "AcquisitionFunction"
            )
        return acq_func

    def dataframe_to_torch(
        self, data: pd.DataFrame, use_transformed=True, keys=None
    ) -> Dict:
        data = self.dataframe_to_numpy(data, use_transformed, keys)
        for name, val in data.items():
            data[name] = torch.tensor(data[name], **self.tkwargs)
        return data

    def _generate(self, data) -> pd.DataFrame:
        """
        Generate datapoints for sampling using an acquisition function and a model
        """
        # save dataframe of most recent generate call for internal use
        self._data = data

        # create model from data
        self.model = self.create_model(data)

        # get acq_function
        self.acq_func = self.get_acqf(self.model, **self.acquisition_function_options)

        # get candidates
        candidates = self._optimize_acq(self.acq_func)

        candidates = candidates.detach().cpu().numpy()
        return untransform_x(self.numpy_to_dataframe(candidates), self.vocs)

    def create_model(self, data, use_transformed=True):
        # get valid data from dataframe and convert to torch tensors
        # + do normalization required by bototrch models
        valid_df = data.loc[data["status"] == "done"]

        # check to make sure there is some data
        if len(valid_df) == 0:
            raise RuntimeError("no data to create GP model")

        if use_transformed:
            train_data = self.dataframe_to_torch(valid_df)

            # negate objective values -> bototrch assumes maximization
            train_data["Y"] = -train_data["Y"]
        else:
            train_data = self.dataframe_to_torch(valid_df, False)

        # create and train model
        return self.create_model_f(train_data, vocs=self.vocs)

    def _optimize_acq(self, acq_func) -> torch.Tensor:

        # optimize
        self.optimization_options["raw_samples"] = (
            self.optimization_options["raw_samples"] or 512
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self._n_samples,
            num_restarts=self.optimization_options.get("num_restarts", 20),
            **self.optimization_options,
        )

        return candidates
