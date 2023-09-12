import yaml


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data
