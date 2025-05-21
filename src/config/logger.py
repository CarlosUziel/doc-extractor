import sys
from pathlib import Path

from loguru import logger

# Define the log file path relative to the project root
LOG_FILE_PATH = Path(__file__).parents[2] / ".logs" / "app.log"

# Ensure the log directory exists
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)
logger.add(
    LOG_FILE_PATH,
    rotation="10 MB",  # Rotate log file when it reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress rotated files
    level="DEBUG",  # Log DEBUG level and above to file
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,  # Asynchronous logging
)

# Export the configured logger instance
log = logger
