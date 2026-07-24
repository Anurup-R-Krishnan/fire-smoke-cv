"""Classical ML: classical computer vision and conventional ML baselines."""

from .config import ClassicalMLConfig, load_config
from .pipeline import run_pipeline

__all__ = ["ClassicalMLConfig", "load_config", "run_pipeline"]
