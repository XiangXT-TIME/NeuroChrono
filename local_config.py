import os
import yaml

_config = None
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    global _config
    if _config is not None:
        return _config
    config_path = os.path.join(_PROJECT_ROOT, "local_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"local_config.yaml not found at {config_path}. "
            f"Copy local_config.example.yaml to local_config.yaml and edit paths for your server."
        )
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    return _config


def get_paths():
    """Return the paths dict from local_config.yaml."""
    return _load_config()["paths"]
