import os
import logging
from dotenv import load_dotenv

load_dotenv()

def setup_logger(name:str) -> logging.Logger:

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d  -- %(name)s -- | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]  # Console only
    )
    
    return logging.getLogger(name)


