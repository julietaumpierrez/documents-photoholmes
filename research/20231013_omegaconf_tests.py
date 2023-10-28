from typing import Any, Dict, Optional

from omegaconf import OmegaConf


def from_config(config: Optional[str | Dict[str, Any]]):
    """
    Instantiate the model from configuration dictionary or yaml.

    Params:
        config: path to the yaml configuration or a dictionary with
                the parameters for the model.
    """
    if isinstance(config, str):
        config = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)
    print(config.threshold)
    print(config)
    return


from_config("src/photoholmes/models/DQ/config.yaml")
