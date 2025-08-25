
def get_config_path(name: str):
    from pathlib import Path
    if name.endswith(".yaml"):
        return Path(__file__).parent / name
    else:
        return Path(__file__).parent / f"{name}.yaml"