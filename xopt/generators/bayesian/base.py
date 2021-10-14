from abc import ABC

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim.optimize import optimize_acqf

from ..generator import ContinuousGenerator
from .models.models import create_model
from ...vocs_tools import get_bounds
from typing import Dict, Callable
import pandas as pd
from ...utils import check_and_fill_defaults
from ...tools import get_function_defaults

import logging

logger = logging.getLogger(__name__)


class BayesianGenerator(ContinuousGenerator, ABC):
    def __init__(self,
                 vocs: Dict,
                 acqisition_function: Callable,
                 acquisition_options: Dict = None,
                 optimize_options: Dict = None
                 ):
        """
        General Bayesian optimization generator
        NOTE: we assume maximization for all acquisition functions
        """
        super(BayesianGenerator, self).__init__(vocs)
        self.model = None
        self.acqisition_function = acqisition_function
        self.acqisition_function_options = acquisition_options or {}

        # get optimization kwargs defaults
        optimization_defaults = get_function_defaults(optimize_acqf)

        self.optimization_options = check_and_fill_defaults(optimize_options,
                                                            optimization_defaults)

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

    def _generate(self, data) -> pd.DataFrame:
        """
        Generate datapoints for sampling using an acquisition function and a model
        """

        # get valid data from dataframe and convert to torch tensors
        valid_df = data.loc[data['status'] == 'done']
        train_data = self.dataframe_to_numpy(valid_df)
        for name, val in train_data.items():
            train_data[name] = torch.tensor(train_data[name], **self.tkwargs)

        # negate objective values -> bototrch assumes maximization
        train_data['Y'] = -train_data['Y']

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
            acq_function=acq_func,
            bounds=bounds,
            q=self._n_samples,
            num_restarts=self.optimization_options.get('num_restarts', 20),
            raw_samples=self.optimization_options.get('raw_samples', 512),
            **self.optimization_options
        )

        candidates = candidates.detach().cpu().numpy()
        return self.numpy_to_dataframe(candidates)
