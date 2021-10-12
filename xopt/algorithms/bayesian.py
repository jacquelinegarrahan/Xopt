from abc import ABC

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim.optimize import optimize_acqf

from .algorithm import Algorithm
from ..bayesian.models.models import create_model
from ..vocs_tools import get_bounds
from typing import Dict, Callable
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class BayesianAlgorithm(Algorithm, ABC):
    def __init__(self,
                 vocs: Dict,
                 acqisition_function: Callable,
                 acquisition_options: Dict = None,
                 optimize_options: Dict = None
                 ):
        """
        General Bayesian optimization algorithm
        """
        super(BayesianAlgorithm, self).__init__(vocs)
        self.model = None
        self.acqisition_function = acqisition_function
        self.acqisition_function_options = acquisition_options or {}
        self.optimization_options = {'num_restarts': 20,
                                     'raw_samples': 200}
        self.optimization_options.update(optimize_options)

        self.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

        # set up gpu if requested
        use_gpu = acquisition_options.get('use_gpu', False)
        if use_gpu:
            if torch.cuda.is_available():
                self.tkwargs["device"] = torch.device("cuda")
                logger.info(
                    f"using gpu device "
                    f'{torch.cuda.get_device_name(self.tkwargs["device"])}'
                )
            else:
                logger.warning("gpu requested but not found, using cpu")

        self.n_calls = 0

    def generate(self, data) -> pd.DataFrame:
        """
        Generate datapoints for sampling using an acquisition function and a model
        """

        # get data from dataframe and convert to torch tensors
        train_data = self.dataframe_to_numpy(data)
        for name, val in train_data.items():
            train_data[name] = torch.tensor(train_data[name], **self.tkwargs)

        # create and train model
        self.model = create_model(train_data, vocs=self.vocs)

        # optimize the acquisition function
        bounds = torch.tensor(get_bounds(self.vocs), **self.tkwargs)

        # set up acquisition function object
        acq_func = self.acqisition_function(self.model,
                                            **self.acqisition_function_options)

        if not isinstance(acq_func, AcquisitionFunction):
            raise RuntimeError(
                "callable `acqisition_function` does not return type "
                "AcquisitionFunction"
            )

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func, bounds=bounds, **self.optimization_options
        )

        candidates = candidates.detach().cpu().numpy()
        self.n_calls += 1
        return self.numpy_to_dataframe(candidates)


class UpperConfidenceBound(BayesianAlgorithm):
    def __init__(self, vocs, n_steps=1, q=1, beta=2.0, **kwargs):
        acquisition_options = {'beta': beta}
        optimize_options = {'q': q,
                            **kwargs}
        self.n_steps = n_steps
        super(UpperConfidenceBound, self).__init__(vocs,
                                                   qUpperConfidenceBound,
                                                   acquisition_options,
                                                   optimize_options)

    def is_terminated(self):
        return self.n_calls >= self.n_steps
