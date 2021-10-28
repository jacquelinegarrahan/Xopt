from . import _version

__version__ = _version.get_versions()["version"]

from xopt.log import configure_logger
from .tools import xopt_logo
from .xopt import Xopt


def output_notebook(level="INFO"):
    """
    Redirects logging to stdout for use in Jupyter notebooks
    """
    configure_logger(level=level)
