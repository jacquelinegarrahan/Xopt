import numpy as np
import pandas as pd

from xopt.algorithms.generators import transform_data


class TestTransforms:
    test_vocs = {
        "variables": {"x1": [0, 10], "x2": [-10, 0]},
        "objectives": {"y1": "MINIMIZE", "y2": "MAXIMIZE"},
        "constraints": {},
    }

    def test_transform(self):
        data = np.random.rand(10, 4) * 10 - 5.0
        df = pd.DataFrame(
            data,
            columns=list(self.test_vocs["variables"])
            + list(self.test_vocs["objectives"]),
        )

        tdata = transform_data(df, self.test_vocs)
        for key in list(self.test_vocs["objectives"]):
            assert np.isclose(tdata[key + "_t"].mean(), 0.0)
            assert np.isclose(tdata[key + "_t"].std(), 1.0)
