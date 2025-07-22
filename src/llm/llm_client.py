"""This module initializes the Large Language Model (LLM) client using 
configuration parameters from environment variables and loads the relevant
 prompt templates. These templates are used to guide the LLM in selecting 
 appropriate function calls and generating user responses based on retrieved data. """

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.logger_config import setup_logger
from langchain_core.output_parsers import StrOutputParser 
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain.chains import LLMChain
from src.utils.errors import LlmClientError

load_dotenv()
logger = setup_logger(__name__)



class LlmClient:
    def __init__(self) -> None:
        """Initializes the LLM client """
        try:
            self._initialize_components()
            self._load_prompt_templates()
            self._config_gemini()
            self._initialize_LLMChain()
            logger.info("LLM client initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing the LLM client: {e}", exc_info=True)
            raise LlmClientError("Failed to initialize LLM client") from e

    def _initialize_components(self) -> None:
        """Initializes the LLM client components with error handling."""

        try:
            self.output_parser = StrOutputParser()
            
            self.llm_gemini = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.5,
                    client=genai,
                    max_output_tokens=1024,
                    convert_system_message_to_human=True,
                    client_options={},
                    transport=None
                )
            
            self.llm_biomistral = ChatOllama(
                model="biomistral",
                temperature=0.5

            )
            
            self.memory = ConversationSummaryBufferMemory(
            llm=self.llm_biomistral,
            memory_key="chat_history", 
            max_token_limit=1000,
            return_messages=False
         )
                    
            logger.info("LLM client components initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLM client components: {e}", exc_info=True)
            raise LlmClientError("Component initialization failed") from e
    def _load_prompt_templates(self):
        """Load prompt templates with clear error handling."""
        base_path = Path(__file__).parent.parent / "prompts"
        
        try:
            # Load function calling prompt
            function_calling_path = base_path / "function_call_prompt.txt"
            with open(function_calling_path, "r") as f:
                function_calling_content = f.read().strip()
            
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
            self.function_calling_prompt_template = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=function_calling_content
         )
            
            self.answer_generation_prompt_template = PromptTemplate(
            input_variables=["chat_history", "question", "formatted_results"],
            template=answer_generation_content
            )
        except Exception as e:
            logger.error(f"Error creating prompt templates: {e}", exc_info=True)
            raise LlmClientError("Prompt template creation failed") from e

    def _initialize_LLMChain(self):
        """Initializes the LLMChain with the function calling prompt template."""
        try:
           
            self.function_calling_chain = LLMChain(
                llm=self.llm_gemini,
                prompt=self.function_calling_prompt_template,
                output_parser=self.output_parser,
                memory=self.memory
            )
            self.answer_generation_chain = LLMChain(
                llm=self.llm_biomistral,
                prompt=self.answer_generation_prompt_template,
                output_parser=self.output_parser
            )
            logger.info("LLMChain initialized successfully.")
        except Exception as e:
            logger.error(f"An error occurred initializing LLMChain: {e}", exc_info=True)
            raise LlmClientError("Failed to initialize LLMChain") from e

    def _config_gemini(self):
        try:
            google_api = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=google_api)

        except Exception as e:
            logger.error(f"An error occurred configuring Gemini: {e}", exc_info=True)
            raise LlmClientError("Failed to configure Gemini") from e
