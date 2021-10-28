import numpy as np
import torch

from xopt.algorithms.generators.bayesian.models.models import create_model


class TestModelCreation:
    vocs = {"variables": {"x1": [0, 1], "x2": [0, 1], "x3": [0, 1]}}

    def test_create_model(self):
        train_x = torch.rand(5, 3)
        train_y = torch.rand(5, 2)
        train_c = torch.rand(5, 4)

        train_data = {"X": train_x, "Y": train_y, "C": train_c}
        model = create_model(train_data, vocs=self.vocs)

        train_y_nan = train_y.clone()
        train_y_nan[0][1] = np.nan
        train_data = {"X": train_x, "Y": train_y, "C": train_c}

        model = create_model(train_data, vocs=self.vocs)
