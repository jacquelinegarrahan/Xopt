import json
import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Union

import pandas as pd

from ..evaluators.evaluator import Evaluator
from ..generators.generator import Generator, ContinuousGenerator
from .control_flow import KNOWN_CONTROL_FLOW

logger = logging.getLogger(__name__)


class Algorithm(ABC):
    def __init__(
        self,
        vocs: Dict,
        evaluator: Evaluator,
        generator: Union[Generator, ContinuousGenerator],
        output_path: str = "",
        restart_file: str = None,
        control_flow: str = "batched",
        n_initial_samples: int = 1,
    ):
        self.vocs = vocs
        self.evaluator = evaluator
        self.generator = generator
        self.output_path = output_path
        self.control_flow = control_flow
        self.n_initial_samples = n_initial_samples

        if restart_file is not None:
            self.load_data(restart_file)
        else:
            self.data = None

    def run(self):
        """
        run algorithm
        """
        return KNOWN_CONTROL_FLOW[self.control_flow](self)

    def save_data(self):
        """
        Save data and config to a json file
        """
        # start with config info
        info = deepcopy(self.vocs)

        # add data
        if self.data is not None:
            info.update({"results": json.loads(self.data.to_json())})

        logger.debug("saving data to file")
        with open(os.path.join(self.output_path, "results.json"), "w") as outfile:
            json.dump(info, outfile, default=serializable)

    def load_data(self, fname):
        """
        load data from file, assumes a data file in json format, looks for `results` key
        """
        with open(fname, "r") as infile:
            data_dict = json.load(infile)

        self.data = pd.DataFrame(data_dict["results"])


# define a default json serializer
def serializable(obj):
    return obj.__dict__
