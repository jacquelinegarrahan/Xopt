from abc import ABC, abstractmethod
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
                 evaluator: Evaluator,
                 generator: Union[Generator, ContinuousGenerator],
                 output_path: str = ''):
        self.config = config
        self.vocs = config['vocs']
        self.evaluator = evaluator
        self.generator = generator
        self.output_path = output_path

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
            info.update({'results': self.data.to_json()})

        with open(os.path.join(self.output_path, 'results.json'), 'w') as outfile:
            json.dump(info, outfile)

