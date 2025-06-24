"""This module processes raw data, generates relevant inferences, 
and prepares the results for storage in the processed data directory, 
facilitating subsequent insertion into the Neo4j database."""

import os
from dotenv import load_dotenv
from utils.logger_config import setup_logger

load_dotenv()

logger = setup_logger(__name__)

logger.debug("from the network.py module")


