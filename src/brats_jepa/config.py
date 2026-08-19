from pathlib import Path
from typing import Any

import yaml

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard directory locations
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "2d_slices"

CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_directories():
    """Creates required project output directories if they do not exist."""
    for directory in [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
