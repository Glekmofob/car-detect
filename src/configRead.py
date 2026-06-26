import yaml


def LoadConfig(filePath = "config.yaml"):
    with open(filePath, "r") as f:
        config = yaml.safe_load(f)
    return config
