from typing import List

import botorch
import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.models import ModelListGP
from botorch.models.model import Model
from torch import Tensor


# define classes that combine acquisition functions

def split_keys(vocs, nominal_quality_parameters):
    target_x_keys = []
    quality_x_keys = []
    for key in vocs['variables']:
        quality_x_keys += [key]
        if key not in nominal_quality_parameters:
            target_x_keys += [key]
    return target_x_keys, quality_x_keys


class PosteriorUncertainty(botorch.acquisition.analytic.AcquisitionFunction):
    def __init__(self, model, **kwargs):
        super(PosteriorUncertainty, self).__init__(model)
        self._ucb = UpperConfidenceBound(model, 1e8, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self._ucb(X) / 1e8


class QualityAwareExploration(botorch.acquisition.acquisition.AcquisitionFunction):
    def __init__(self,
                 model: ModelListGP,
                 target_parameter_indicies: List,
                 beta: float = 2.0,
                 ):
        """
        Acquisition function to do quality aware optimization. Free parameters are
        divided into two groups, target_parameters and quality_parameters. We assume
        that our target function depends strongly on target_parameters and weakly on
        quality_parameters and that the measurement quality can depend strongly on
        both sets of parameters. As such, we only wish to maximize model uncertainty
        with respect to target_parameters.

        For best results the quality measurement should be normalized to the range [
        0,1]. This is not checked.

        Parameters
        ----------
        model : botorch.models.ModelListGP
            Bayesian models that predict two values, the first is the target
            measurement that should be explored and the second is the quality
            measurement. The target model should accept an input tensor of shape
            n x t x len(target_parameter_indicies).

        target_parameter_indicies : List
            Indicies of input tensor that correspond to target function inputs

        beta : float
            Beta parameter of Upper Confidence Bound optimization used for quality
            optimization

        """

        super().__init__(model)

        # check to make sure that the target function model has correct input dimensions
        if model.models[0].train_inputs[0].shape[-1] != len(target_parameter_indicies):
            raise RuntimeError('target model feature dimension does not match the '
                               'number of target parameters')

        self.target_parameter_indicies = target_parameter_indicies
        self.target_acq = PosteriorUncertainty(model.models[0])
        self.quality_acq = UpperConfidenceBound(model.models[1], beta)

    def forward(self, X: Tensor) -> Tensor:
        # calculate the posterior uncertainty where the quality parameters are at
        # their nominal values
        pos = self.get_target_acq(X)
        qual = self.get_qual_acq(X)

        return pos * qual

    def get_target_acq(self, X: Tensor) -> Tensor:
        pos = self.target_acq.forward(X[..., self.target_parameter_indicies])
        return pos

    def get_qual_acq(self, X: Tensor) -> Tensor:
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
