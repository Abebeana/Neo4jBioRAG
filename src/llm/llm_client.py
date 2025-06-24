"""This module initializes the Large Language Model (LLM) client using 
configuration parameters from environment variables and loads the relevant
 prompt templates. These templates are used to guide the LLM in selecting 
 appropriate function calls and generating user responses based on retrieved data. """

import os
from dotenv import load_dotenv
from utils.logger_config import setup_logger

load_dotenv()

logger = setup_logger(__name__)

logger.debug("from the llm_client.py file")


