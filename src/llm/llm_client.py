"""
This module initializes the Large Language Model (LLM) client.

It handles the configuration and setup of all components related to the LLM,
including:
- Loading configuration from environment variables.
- Initializing the Google Gemini model.
- Setting up conversational memory.
- Loading prompt templates for various tasks.
- Creating a dedicated LLMChain for generating natural language answers.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain

from src.utils.logger_config import setup_logger
from src.utils.errors import LlmClientError

# Load environment variables from the .env file
load_dotenv()

# Initialize the logger for this module
logger = setup_logger(__name__)


class LlmClient:
    """
    A client to manage all interactions with the Large Language Model.

    This class encapsulates the setup for the LLM, memory, and the answer
    generation chain, providing a single point of access for LLM-related
    operations.
    """
    def __init__(self, database) -> None:
        """
        Initializes the LLM client by setting up all necessary components.

        Args:
            database: An active database connection instance.

        Raises:
            LlmClientError: If any part of the initialization fails.
        """
        try:
            self.database = database
            self._config_gemini()
            self._initialize_components()
            self._load_prompt_templates()
            self._initialize_LLMChain()
            logger.info("LLM client initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred during LLM client initialization: {e}", exc_info=True)
            raise LlmClientError("Failed to initialize LLM client") from e

    def _config_gemini(self):
        """Configures the Google Generative AI client with an API key from .env."""
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            genai.configure(api_key=google_api_key)
            logger.info("Google Gemini client configured successfully.")
        except Exception as e:
            logger.error(f"An error occurred configuring Gemini: {e}", exc_info=True)
            raise LlmClientError("Failed to configure Gemini") from e

    def _initialize_components(self) -> None:
        """Initializes the core LLM and memory components."""
        try:
            # Load LLM parameters from environment variables with defaults
            model_name = os.getenv("LLM_MODEL_NAME", "gemini-pro")
            temperature = float(os.getenv("LLM_TEMPERATURE", 0.1))
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", 1024))

            # Initialize the ChatGoogleGenerativeAI model
            self.llm_gemini = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                client=genai,
                max_output_tokens=max_tokens,
                convert_system_message_to_human=True,
                client_options={},
                transport=None,
            )

            # Initialize conversation memory to maintain context across turns
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm_gemini,
                max_token_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", 1000)),
                memory_key="chat_history", # Key used in prompts
                input_key="input",
                output_key="output",
                return_messages=True
            )
            self.llm_ollama = ChatOllama(
                model="cniongolo/biomistral:latest",
                temperature=temperature,
            )
            logger.info("LLM and memory components initialized.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLM components: {e}", exc_info=True)
            raise LlmClientError("Component initialization failed") from e

    def _load_prompt_templates(self):
        """Loads prompt templates from the /prompts directory."""
        base_path = Path(__file__).parent.parent / "prompts"
        try:
            # Load the template for generating the final natural language answer
            answer_generation_path = base_path / "answer_generation_prompt.txt"
            with open(answer_generation_path, "r") as f:
                answer_generation_content = f.read().strip()
            
            self.answer_generation_prompt_template = PromptTemplate(
                input_variables=["input", "formatted_results"],
                template=answer_generation_content
            )
            logger.info("Prompt templates loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Prompt file not found: {e.filename}", exc_info=True)
            raise LlmClientError(f"Prompt file not found: {e.filename}") from e
        except Exception as e:
            logger.error(f"Error loading or creating prompt templates: {e}", exc_info=True)
            raise LlmClientError("Failed to load prompt templates") from e

    def _initialize_LLMChain(self):
        """Initializes the LLMChain for answer generation."""
        try:
            # This chain combines the answer generation prompt with the LLM
            self.answer_generation_chain = LLMChain(
                llm=self.llm_ollama,
                prompt=self.answer_generation_prompt_template,
            )
            logger.info("Answer generation LLMChain initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLMChain: {e}", exc_info=True)
            raise LlmClientError("Failed to initialize LLMChain") from e
