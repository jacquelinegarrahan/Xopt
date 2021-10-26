import botorch
import torch
from torch import Tensor
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition.analytic import PosteriorMean
from copy import deepcopy
# define classes that combine acquisition functions


class PosteriorUncertainty(botorch.acquisition.analytic.AcquisitionFunction):
    def __init__(self, model, **kwargs):
        super(PosteriorUncertainty, self).__init__(model)
        self._ucb = UpperConfidenceBound(model, 1e8, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self._ucb(X) / 1e8


class QualityAwareExploration(botorch.acquisition.acquisition.AcquisitionFunction):
    """
    Acquisition function to do quality aware optimization. Free parameters are
    divided into two groups, target_parameters and quality_parameters. We assume that
    our target function depends strongly on target_parameters and weakly on
    quality_parameters and that the measurement quality can depend strongly on both sets
    of parameters. As such, we only wish to maximize model uncertainty with respect to
    target_parameters, fixing quality_parameters at nominal values.


    """

    def __init__(self,
                 model,
                 nominal_quality_parameters,
                 target_idx=0,
                 quality_idx=1,
                 beta=2.0,
                 tkwargs=None
                 ):
        super().__init__(model)
        self.nominal = nominal_quality_parameters

        model_copy = deepcopy(model)

        # modify model copy such that the length scale of the target function with
        # respect to the quality parameters is very long (assumes normalization)
        model_lengthscales = model_copy.covar_module.base_kernel.lengthscale
        for ele in nominal_quality_parameters.keys():
            model_lengthscales[0, target_idx, ele] = 100.0
        model_copy.covar_module.base_kernel.lengthscale = model_lengthscales

        tkwargs = tkwargs or {}
        w1 = torch.zeros(model_copy.num_outputs, **tkwargs)
        w1[target_idx] = 1.0
        w2 = torch.zeros(model_copy.num_outputs, **tkwargs)
        w2[quality_idx] = 1.0
        self.target_acq = PosteriorUncertainty(model_copy,
                                               objective=ScalarizedObjective(w1))

        self.quality_acq = PosteriorMean(model_copy,
                                         objective=ScalarizedObjective(w2))

    def forward(self, X: Tensor) -> Tensor:
        # calculate the posterior uncertainty where the quality parameters are at
        # their nominal values
        pos = self.get_target_acq(X)
        qual = self.get_qual_acq(X)

        return pos * qual

    def get_target_acq(self,  X: Tensor) -> Tensor:
        X_target_eval = X.clone()
        for idx, value in self.nominal.items():
            X_target_eval[..., idx] = value
        pos = self.target_acq.forward(X_target_eval)
        return pos

    def get_qual_acq(self,  X: Tensor) -> Tensor:
        return torch.log(1 + torch.exp(self.quality_acq.forward(X)))


class MultiplyAcquisitionFunction(botorch.acquisition.acquisition.AcquisitionFunction):
    def __init__(self, model, acquisition_functions):
        """
        Acquisition function class that combines several seperate acquisition functions
        together by multiplying them

        Arguments
        ---------
        acquisition_functions : list
            List of acquisition functions to multiply together

        """

        super().__init__(model)

        self.acqisition_functions = acquisition_functions

    def forward(self, X):
        value = torch.ones(X.shape[0])

        for function in self.acqisition_functions:
            multiplier = function.forward(X)
            value = value * multiplier

        return value
