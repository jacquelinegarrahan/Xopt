import pandas as pd
from typing import Dict
from xopt.vocs_tools import get_bounds
import numpy as np


def transform_data(dataframe: pd.DataFrame,
                   vocs: Dict):
    """
    creates a new dataframe that has the following properties, according to vocs:
    - nomalizes variables based on bounds
    - flips sign of objectives if target is 'MAXIMIZE'
    - modifies costraint values; feasible if less than or equal to zero
    """
    new_df = dataframe.copy()

    # normalize x data based on bounds
    bounds = get_bounds(vocs)
    for ii, name in enumerate(vocs['variables']):
        new_df[name + '_t'] = \
            (new_df[name] - bounds[0, ii]) / (bounds[1, ii] - bounds[0, ii])

    # flip sign of objectives 'MAXIMIZE'
    for name in vocs['objectives']:
        if vocs['objectives'][name] == 'MAXIMIZE':
            new_df[name + '_t'] = -new_df[name]
        else:
            new_df[name + '_t'] = new_df[name]

    # modify constraint values
    if vocs['constraints'] is not None:
        for name in vocs['constraints']:
            if vocs['constraints'][name][0] == 'GREATER_THAN':
                new_df[name + '_t'] = vocs['constraints'][name][1] - new_df[name]

            elif vocs['constraints'][name][0] == 'LESS_THAN':
                new_df[name + '_t'] = -(vocs['constraints'][name][1] - new_df[name])

            else:
                raise ValueError(f"{vocs['constraints'][name][0]} not accepted")

        # add feasibility metric
        ndata = new_df[[ele + '_t' for ele in vocs['constraints']]].to_numpy()
        fdata = ndata <= 0
        new_df[[ele + '_f' for ele in vocs['constraints']]] = fdata
        new_df['feas'] = np.all(fdata, axis=1)

    return new_df


