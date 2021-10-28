from typing import Dict
from copy import deepcopy
import logging
import warnings

logger = logging.getLogger(__name__)


def reformat_config(config: Dict) -> Dict:
    """
    Reformat old config files to enable their use for new versions of xopt.
    Raise a bunch of warnings so it annoys people into updating their config files

    Parameters
    ----------
    config: Dict
        Old config file to be checked

    Returns
    -------
    new_config: Dict
        Updated config file

    """
    # copy config
    new_config = deepcopy(config)

    # check xopt
    if "algorithm" in new_config["xopt"]:
        warnings.warn("`algorithm` keyword no longer allowed in xopt config")
        del new_config["xopt"]["algorithm"]

    if "verbose" in new_config["xopt"]:
        warnings.warn("`verbose` keyword no longer allowed in xopt config")
        del new_config["xopt"]["verbose"]

    # check simulation
    if "simulation" in new_config:
        new_config["evaluate"] = new_config["simulation"]
        warnings.warn(
            "`simulation` keyword no longer allowed in evaluate config, moving to "
            "`evaluate`"
        )
        del new_config["simulation"]

    if "templates" in new_config["evaluate"]:
        warnings.warn(
            "`templates` keyword no longer allowed in evaluate config, "
            "moving to `options`"
        )
        try:
            new_config["evaluate"]["options"].update(
                {"templates": new_config["evaluate"]["templates"]}
            )
        except KeyError:
            new_config["evaluate"]["options"] = {
                "templates": new_config["evaluate"]["templates"]
            }

        del new_config["evaluate"]["templates"]

    # check vocs
    for ele in ["name", "description", "simulation"]:
        if ele in new_config["vocs"]:
            logger.warning(
                f"`{ele}` keyword no longer allowed in vocs config, removing"
            )
            del new_config["vocs"][ele]

    # move templates to simulation
    if "templates" in new_config["vocs"]:
        logger.warning(
            "`templates` keyword no longer allowed in vocs config, "
            "moving to evaluate `options`"
        )
        try:
            new_config["evaluate"]["options"].update(
                {"templates": new_config["vocs"]["templates"]}
            )
        except KeyError:
            new_config["evaluate"]["options"] = {
                "templates": new_config["vocs"]["templates"]
            }

        del new_config["vocs"]["templates"]

    if "generator" in new_config:
        warnings.warn(
            "`generator` keyword no longer allowed in xopt config, "
            "replacing with algorithm"
        )

        new_config["algorithm"] = new_config["generator"]
        del new_config["generator"]

    try:
        if (
            new_config["algorithm"]["name"] is not None
            and new_config["algorithm"]["function"] is not None
        ):
            warnings.warn(
                "can only specify algorithm name OR function, deleting " "function"
            )
            del new_config["algorithm"]["function"]
    except KeyError:
        pass

    # modify function call in evaluate
    if "evaluate" in new_config["evaluate"]:
        logger.warning(
            "`evaluate key in evaluate no longer allowed, replacing with "
            "`function` keyword"
        )
        new_config["evaluate"]["function"] = new_config["evaluate"]["evaluate"]
        del new_config["evaluate"]["evaluate"]

    return new_config
