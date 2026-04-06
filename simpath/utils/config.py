import yaml
from pathlib import Path

def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged
