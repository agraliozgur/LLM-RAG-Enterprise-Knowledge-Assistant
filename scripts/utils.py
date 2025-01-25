# FILE: utils.py

import os
import yaml
import logging
import torch

def load_config(config_path: str = None) -> dict:
    """
    Load the project configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the YAML config file.
                                     If not provided, defaults to 
                                     '../config/project_settings.yaml'.
    Returns:
        dict: Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = os.path.join("../config", "project_settings.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str = None, level=logging.INFO) -> logging.Logger:
    """
    Create a logger with a specified name and logging level.

    Args:
        name (str): The logger name.
        level (int): The logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name if name else __name__)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    ch.setFormatter(formatter)

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        logger.addHandler(ch)

    return logger


def get_device() -> str:
    """
    Detect the best available torch device: CUDA, MPS, or CPU.
    Returns a device string usable by transformers pipelines.

    Returns:
        str: A device string, e.g. "cuda:0", "mps", or "cpu".
    """
    # CUDA?
    if torch.cuda.is_available():
        return "cuda:0"
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # Default CPU
    return "cpu"


def save_model(model, tokenizer, output_dir: str):
    """
    Save a fine-tuned model and tokenizer locally for future reuse.

    Args:
        model: HuggingFace model object
        tokenizer: HuggingFace tokenizer object
        output_dir (str): Directory to save the model and tokenizer.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
