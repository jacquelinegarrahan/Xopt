from abc import ABC, abstractmethod

import pandas as pd

from ..generators.generator import Generator, ContinuousGenerator
from ..evaluators.evaluator import Evaluator
from typing import Dict, Union
from copy import deepcopy
import os
import json

import logging

logger = logging.getLogger(__name__)


class Algorithm(ABC):
    def __init__(self,
                 config: Dict,
                 vocs: Dict,
                 evaluator: Evaluator,
                 generator: Union[Generator, ContinuousGenerator],
                 output_path: str = '',
                 restart_file: str = None):
        self.config = config
        self.vocs = vocs
        self.evaluator = evaluator
        self.generator = generator
        self.output_path = output_path

        if restart_file is not None:
            self.load_data(restart_file)
        else:
            self.data = None

    @abstractmethod
    def run(self):
        """
        run routine
        """
        pass

    def save_data(self):
        """
        Save data and config to a json file
        """
        # start with config info
        info = deepcopy(self.config)

        # add data
        if self.data is not None:
            info.update({'results': json.loads(self.data.to_json())})

        logger.debug('saving data to file')
        with open(os.path.join(self.output_path, 'results.json'), 'w') as outfile:
            json.dump(info, outfile, default=serializable)

    def load_data(self, fname):
        """
        load data from file, assumes a data file in json format, looks for `results` key
        """
        with open(fname, 'r') as infile:
            data_dict = json.load(infile)

        self.data = pd.DataFrame(data_dict['results'])


# define a default json serializer
def serializable(obj):
    return obj.__dict__