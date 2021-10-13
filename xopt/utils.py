import pandas as pd
from typing import Dict


class BadDataError(ValueError):
    pass


class BadFunctionError(ValueError):
    pass


def check_dataframe(df: pd.DataFrame, vocs: Dict):
    """verify dataframe has the minimum column names according to vocs"""
    if not set(list(vocs['variables']) +
               list(vocs['objectives']) +
               list(vocs['constraints'] or {}) +
               ['status']).issubset(set(df.keys())):
        raise BadDataError(
            'data is missing needed columns, is there valid data? is '
            'the dataframe formatted correctly?')
