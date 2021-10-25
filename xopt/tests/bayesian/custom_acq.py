from botorch.acquisition.analytic import UpperConfidenceBound

def acq(model):
    return UpperConfidenceBound(model, 2.0)
