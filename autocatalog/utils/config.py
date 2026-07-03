from pathlib import Path
import yaml

def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}