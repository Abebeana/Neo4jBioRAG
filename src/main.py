import os
from dotenv import load_dotenv
from utils.logger_config import setup_logger

# -----------------------------

from database import neo4j_database
from dataprocessing import data_inference
from llm import llm_client
from retriever import retriever


load_dotenv()

logger = setup_logger(__name__)

logger.debug("from main")


