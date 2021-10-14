import pandas as pd
from typing import Dict
from copy import deepcopy


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


def check_and_fill_defaults(input_dict, defaults):
    check_config_against_defaults(input_dict, defaults)
    return fill_defaults(input_dict, defaults)


def fill_defaults(dict1, defaults):
    """
    Fills a dict with defaults in a defaults dict.

    dict1 must only contain keys in defaults.

    deepcopy is necessary!

    """
    for k, v in defaults.items():
        if k not in dict1:
            dict1[k] = deepcopy(v)

    return dict1


def check_config_against_defaults(test_dict, defaults):
    for k in test_dict:
        if k not in defaults:
            raise Exception(
                f'Extraneous key: {k}. Allowable keys: ' + ', '.join(list(defaults)))