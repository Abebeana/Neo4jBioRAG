""" This module handles the storage of processed data in a Neo4j database, 
enabling efficient data accessibility and management."""

import os
from dotenv import load_dotenv
from utils.logger_config import setup_logger

load_dotenv()

logger = setup_logger(__name__)

logger.debug("from neo4jdatabase module")



