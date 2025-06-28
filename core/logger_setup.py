# stat_arb_trader_dhan/core/logger_setup.py

import logging
import os
import sys
from config import settings

def setup_logger(name="StatArbTrader"):
    """Configures and returns a logger instance."""
    logger_instance = logging.getLogger(name)
    
    # Prevent adding handlers multiple times if this function is called again
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    # Set log level from settings
    try:
        log_level_str = settings.LOG_LEVEL.upper()
        level = getattr(logging, log_level_str, logging.INFO)
    except AttributeError:
        level = logging.INFO
    logger_instance.setLevel(level)

    # Define formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger_instance.addHandler(ch)

    # File Handler
    try:
        fh = logging.FileHandler(settings.LOG_FILE_PATH, mode='a')
        fh.setFormatter(formatter)
        logger_instance.addHandler(fh)
    except Exception as e:
        logger_instance.error(f"Failed to set up file logger at {settings.LOG_FILE_PATH}: {e}", exc_info=True)
        
    return logger_instance

# Initialize a global logger instance for easy import
logger = setup_logger()