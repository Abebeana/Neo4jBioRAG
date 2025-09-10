"""
Main entry point for the Gene Regulatory Network Chatbot.

This script initializes the chatbot by:
1. Loading environment variables.
2. Setting up logging.
3. Loading the gene regulatory network data.
4. Initializing the Neo4j database connection.
5. Initializing the Large Language Model (LLM) client.
6. Creating a LangChain agent with access to custom tools.
7. Running an interactive loop to handle user queries.
"""
import os
from dotenv import load_dotenv
from src.utils.logger_config import setup_logger
from src.database.neo4j_database import Neo4jDatabase
from src.networks.Network import Network
from src.llm.llm_client import LlmClient
from src.agents.agent import create_agent
from typing import Any
import textwrap

# Load environment variables from .env file
load_dotenv()
# Initialize the logger for this module
logger = setup_logger(__name__)

def extract_tool_result(result: dict) -> Any:
    """
    Extracts the raw tool output from the agent's intermediate steps.

    The LangChain agent returns a dictionary containing the final output and
    a list of intermediate steps taken. This function inspects these steps
    to find the direct result from the last tool call, bypassing the agent's
    final summarized answer.

    Args:
        result (dict): The dictionary returned by the agent executor.

    Returns:
        any: The raw observation from the last tool call. Returns a fallback
             string if no tool steps are found.
    """
    # Get the list of intermediate steps, default to an empty list if not found
    steps = result.get("intermediate_steps", [])
    
    # Optional: Debugging prints to inspect the agent's process
    # print("%" * 100)
    # print(f"Extracting tool result from {len(steps)} steps")
    # print(f"Result keys: {result.keys()}")
    # print(f"Steps: {steps}")
    # print("%" * 100)

    # If no steps were taken, return the agent's final output as a fallback
    if not steps:
        return result.get('output', 'No output found')
    
    # Iterate through the steps in reverse to get the last tool result first
    for step in reversed(steps):
        # Steps can be a tuple of (AgentAction, observation)
        if isinstance(step, tuple) and len(step) >= 2:
            action, observation = step, step
            return observation  # Return the raw tool output
        # Or steps can be a dictionary with an 'observation' key
        elif isinstance(step, dict):
            observation = step.get("observation")
            if observation:
                return observation
    
    # Fallback if the loop completes without finding a valid observation
    return result.get('output', 'No tool output found in steps')


def main():
    """
    Main function to run the interactive chatbot.
    
    Initializes all components and enters a loop to accept and process user queries.
    """
    # --- 1. Initialization ---
    logger.info("Initializing chatbot components...")
    
    # Load gene network data from the path specified in .env
    raw_data_path = os.getenv("RAW_DATA_PATH", "data/raw/net.json")
    network = Network.load_from_json(raw_data_path)
    logger.info(f"Loaded network with {len(network.GRN)} entries.")

    # Initialize database and LLM clients
    database = Neo4jDatabase()

    # Ask user for confirmation before loading data into the database
    load_data = input("Do you want to load the network data into the database? (yes/no): ")
    if load_data.lower() == 'yes':
        logger.info("Loading network data into the database...")
        database.store_network(network)
        logger.info("Network data loading complete.")
    llm_client = LlmClient(database)


    # Create the agent executor. This is done once to preserve memory across queries.
    agent_executor = create_agent(
        llm_client=llm_client,
        database=database,
    )
    logger.info("Chatbot initialized successfully. Ready for queries.")

    # --- 2. Interactive Query Loop ---
    while True:
        # Prompt the user for input
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            database.close()  # Gracefully close the database connection
            print("Exiting...")
            break
        
        try:
            # --- 3. Agent Execution ---
            # Invoke the agent with the user's query
            result = agent_executor.invoke({"input": query})
            
            # Extract the direct tool result, bypassing the agent's summary
            tool_result = extract_tool_result(result)
            
            print("*" * 50)
            print(f"Raw Database Result: {tool_result}")
            print("*" * 50)
            
            # --- 4. Answer Generation ---
            # Use a separate chain to generate a natural language response from the tool result
            if isinstance(tool_result, (dict, tuple)) and tool_result != 'No output found':
                formatted_results = str(tool_result)
                try:
                    # Invoke the answer generation chain
                    answer_result = llm_client.answer_generation_chain.invoke({
                        "input": query,
                        "formatted_results": formatted_results
                    })
                    # The result is a dictionary, get the answer from the 'text' key
                    answer = answer_result.get('text', str(answer_result))
                    wrapped_answer = textwrap.fill(answer, width=80)  # Wrap text at 80 characters

                    print("*" * 50)
                    print(f"Generated Answer:\n{wrapped_answer}")
                    print("*" * 50)
                except Exception as chain_error:
                    logger.error(f"Error in answer generation chain: {chain_error}", exc_info=True)
                    print("Error generating a natural language answer. Displaying raw result as fallback.")
            else:
                # Handle cases where the tool did not return a valid result
                print("No structured database results to process for a natural language answer.")
                
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
