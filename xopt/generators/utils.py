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
    constraints_exist = False

    # iterate through the dataframe columns
    for col_name in new_df.keys():

        # normalize input variables
        if col_name in vocs['variables']:
            bounds = vocs['variables'][col_name]
            new_df[col_name + '_t'] = \
                (new_df[col_name] - bounds[0]) / (bounds[1] - bounds[0])

        # flip sign of objectives 'MAXIMIZE'
        if col_name in vocs['objectives']:
            if vocs['objectives'][col_name] == 'MAXIMIZE':
                new_df[col_name + '_t'] = -new_df[col_name]
            else:
                new_df[col_name + '_t'] = new_df[col_name]

        # modify constraint values
        if col_name in vocs['constraints']:
            constraints_exist = True
            if vocs['constraints'][col_name][0] == 'GREATER_THAN':
                new_df[col_name + '_t'] = vocs['constraints'][col_name][1] - \
                                          new_df[col_name]

            elif vocs['constraints'][col_name][0] == 'LESS_THAN':
                new_df[col_name + '_t'] = -(vocs['constraints'][col_name][1] -
                                            new_df[col_name])

            else:
                raise ValueError(f"{vocs['constraints'][col_name][0]} not accepted")

    # if constraints exist add feasibility metric
    if constraints_exist:
        ndata = new_df[[ele + '_t' for ele in vocs['constraints']]].to_numpy()
        fdata = ndata <= 0
        new_df[[ele + '_f' for ele in vocs['constraints']]] = fdata
        new_df['feas'] = np.all(fdata, axis=1)

    return new_df


