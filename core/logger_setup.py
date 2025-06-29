# core/logger_setup.py (Corrected)

import logging
import os
import sys
import asyncio
from config import settings

# Custom Handler for WebSocket logging
class WebSocketLogHandler(logging.Handler):
    """
    A custom logging handler that sends log records over a WebSocket connection.
    It sends a structured JSON object to the frontend.
    """
    def __init__(self, ws_manager):
        super().__init__()
        self.ws_manager = ws_manager

    def emit(self, record):
        # Create a structured log entry from the record's attributes.
        # This is more robust than trying to format strings in the handler.
        log_entry = {
            "timestamp": record.created,  # UNIX timestamp, flexible for the frontend
            "level": record.levelname,
            "message": record.getMessage(), # The raw, unformatted message
            "module": record.module,
            "lineno": record.lineno,
            "full_formatted_message": self.format(record) # Also send the full string for easy display
        }
        
        # Schedule the broadcast on the event loop. This is thread-safe.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ws_manager.broadcast_log(log_entry))
        except RuntimeError:
            # This can happen if logging occurs when no event loop is running.
            # In our FastAPI app context, this should be rare.
            pass

def setup_logger(name="StatArbTrader", ws_manager=None):
    """Configures and returns a logger instance. Can be enhanced with a WebSocket manager."""
    logger_instance = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    try:
        log_level_str = settings.LOG_LEVEL.upper()
        level = getattr(logging, log_level_str, logging.INFO)
    except AttributeError:
        level = logging.INFO
    logger_instance.setLevel(level)

    # This formatter will be used by all handlers, including our custom one
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger_instance.addHandler(ch)

    # File Handler
    try:
        # Ensure the log directory exists before setting up the handler
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        fh = logging.FileHandler(settings.LOG_FILE_PATH, mode='a')
        fh.setFormatter(formatter)
        logger_instance.addHandler(fh)
    except Exception as e:
        logging.basicConfig()
        logging.error(f"FATAL: Could not set up file logger at {settings.LOG_FILE_PATH}: {e}", exc_info=True)

    # WebSocket Handler (only if the manager is provided)
    if ws_manager:
        wh = WebSocketLogHandler(ws_manager)
        wh.setFormatter(formatter) # The handler will use this to create the 'full_formatted_message'
        logger_instance.addHandler(wh)
        
    return logger_instance

# Initialize a global logger instance.
# This instance will be re-configured by the api_server on startup.
logger = setup_logger()