"""
Configuration loader for the Medical NLP Chatbot.

Loads settings from config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from medical_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def get_env(key: str, default: str | None = None) -> str:
    """
    Retrieve an environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The environment variable value.

    Raises:
        EnvironmentError: If the variable is not set and no default is provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{key}' is not set. "
            f"Please set it in your .env file or system environment."
        )
    return value


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
