"""
This module initializes the Large Language Model (LLM) client using 
configuration parameters from environment variables and loads the relevant
prompt templates. These templates are used to guide the LLM in selecting 
appropriate function calls and generating user responses based on retrieved data. 
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
import google.generativeai as genai

from src.utils.logger_config import setup_logger
from src.utils.errors import LlmClientError

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(__name__)


class LlmClient:
    def __init__(self, database) -> None:
        """Initializes the LLM client."""
        try:
            self.database = database
            self._initialize_components()
            self._load_prompt_templates()
            self._config_gemini()
            self._initialize_LLMChain()
            logger.info("LLM client initialized successfully.")
        except Exception as e:
            logger.error(
                f"An error occurred initializing the LLM client: {e}", exc_info=True
            )
            raise LlmClientError("Failed to initialize LLM client") from e

    def _initialize_components(self) -> None:
        """Initializes the LLM client components with error handling."""
        try:
            self.output_parser = StrOutputParser()
            model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-pro")

            self.llm_gemini = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.5,
                client=genai,
                max_output_tokens=1024,
                convert_system_message_to_human=True,
                client_options={},
                transport=None,
            )

            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm_gemini,
                max_token_limit=1000,
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                return_messages=True
            )

            logger.info("LLM client components initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLM client components: {e}", exc_info=True)
            raise LlmClientError("Component initialization failed") from e

    def _load_prompt_templates(self):
        """Load prompt templates with clear error handling."""
        base_path = Path(__file__).parent.parent / "prompts"

        try:
            # Load answer generation prompt
            answer_generation_path = base_path / "answer_generation_prompt.txt"
            with open(answer_generation_path, "r") as f:
                answer_generation_content = f.read().strip()

        except FileNotFoundError as e:
            logger.error(f"Prompt file not found: {e.filename}", exc_info=True)
            raise LlmClientError(f"Prompt file not found: {e.filename}") from e
        except Exception as e:
            logger.error(f"Error loading prompt files: {e}", exc_info=True)
            raise LlmClientError("Failed to load prompt files") from e

        # Create templates
        try:
            self.answer_generation_prompt_template = PromptTemplate(
                input_variables=["chat_history", "input", "formatted_results"],
                template=answer_generation_content
            )
        except Exception as e:
            logger.error(f"Error creating prompt templates: {e}", exc_info=True)
            raise LlmClientError("Prompt template creation failed") from e

    def _initialize_LLMChain(self):
        """Initializes the LLMChain with the function calling prompt template."""
        try:
            self.answer_generation_chain = LLMChain(
                output_parser=self.output_parser,
                llm=self.llm_gemini,
                prompt=self.answer_generation_prompt_template,
                memory=self.memory,
            )
            logger.info("LLMChain initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLMChain: {e}", exc_info=True)
            raise LlmClientError("Failed to initialize LLMChain") from e

    def _config_gemini(self):
        """Configure Google Gemini client with API key."""
        try:
            google_api = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=google_api)
        except Exception as e:
            logger.error(f"An error occurred configuring Gemini: {e}", exc_info=True)
            raise LlmClientError("Failed to configure Gemini") from e
