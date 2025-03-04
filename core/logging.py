"""
Logging configuration for the application.
"""
import logging

def setup_logger():
    """Configure and return the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Create a logger instance
logger = setup_logger()