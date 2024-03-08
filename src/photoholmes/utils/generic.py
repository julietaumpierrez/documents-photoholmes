from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data
