# src/utils/helpers.py
"""
Common utility helpers for the PhreshPhish project.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: deterministic behaviour (may slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda' if available and requested, otherwise 'cpu'."""
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a dictionary as a pretty-printed JSON file."""
    import json
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)