"""
Logging utilities.
"""
import logging
import sys
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.
    
    Args:
        name (str): Logger name.
        log_file (str, optional): Path to log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
