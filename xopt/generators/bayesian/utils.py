import logging
import time
import traceback
from concurrent.futures import Executor
from typing import Dict, Optional, Tuple, Callable

import torch
from torch import Tensor

from .models.models import create_model


class NoValidResultsError(Exception):
    pass


class UnsupportedError(Exception):
    pass


# Logger
logger = logging.getLogger(__name__)

algorithm_defaults = {
    "n_steps": 30,
    "executor": None,
    "n_initial_samples": 5,
    "custom_model": None,
    "output_path": None,
    "verbose": True,
    "restart_data_file": None,
    "initial_x": None,
    "use_gpu": False,
    "eval_args": None,
}

