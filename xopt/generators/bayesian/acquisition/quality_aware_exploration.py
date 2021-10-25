import botorch
import torch
from torch import Tensor
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedObjective


# define classes that combine acquisition functions


class PosteriorUncertainty(botorch.acquisition.analytic.AnalyticAcquisitionFunction):
    def __init__(self, model, **kwargs):
        super(PosteriorUncertainty, self).__init__(model)
        self._ucb = UpperConfidenceBound(model, 1e8, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self._ucb(X) / 1e8


class QualityAwareExploration(botorch.acquisition.acquisition.AcquisitionFunction):
    def __init__(self, model, target_idx=0, quality_idx=1, beta=2.0):
        super().__init__(model)
        w1 = torch.zeros(model.num_outputs)
        w1[target_idx] = 1.0
        w2 = w1.clone()
        w2[quality_idx] = 1.0
        target_acq = PosteriorUncertainty(model, objective=ScalarizedObjective(w1))

        quality_acq = UpperConfidenceBound(model, beta,
                                           objective=ScalarizedObjective(w2))

        self.total_acq = MultiplyAcquisitionFunction(model, [target_acq, quality_acq])

    def forward(self, X: Tensor) -> Tensor:
        return self.total_acq(X)


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
