import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_dir="logs"):
    """
    Configure logger with file and console output.
    
    Args:
        name (str): Logger name (e.g., 'headlines_scraper')
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Timestamp for unique log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/{name}_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (only INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
