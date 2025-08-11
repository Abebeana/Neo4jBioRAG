"""
This module defines the creation of a LangChain agent responsible for
interpreting user queries and selecting the appropriate database tool.
"""
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agents.gene_tools import get_gene_tools
from src.database.neo4j_database import Neo4jDatabase
from src.llm.llm_client import LlmClient


def create_agent(
   llm_client: LlmClient,
   database: Neo4jDatabase,
) -> AgentExecutor:
    """
    Creates and configures a structured chat agent for gene regulatory networks.

    This agent is designed to function as a "tool router." Its primary role is
    to understand the user's question, select the most appropriate tool from
    the available gene tools, and pass the necessary arguments (like gene_name
    or tf_name) to it. It does not generate the final answer itself.

    Args:
        llm_client (LlmClient): An instance of the LLM client.
        database (Neo4jDatabase): An instance of the Neo4j database connection.

    Returns:
        AgentExecutor: The configured LangChain agent executor, ready to process queries.
    """
    # Get the list of available tools that the agent can use.
    tools = get_gene_tools(database)
    
    # Define the system prompt that instructs the agent on its role and rules.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Your sole responsibility is to select the correct tool and arguments to answer questions about gene regulatory networks.

                Conversation Handling Rules:
                - Always use chat history to resolve missing context. If the user omits the target (e.g., "repressors"), assume they mean the last explicitly mentioned gene or transcription factor.
                - Tolerate minor typos and infer the intended term (e.g., "repreoosrs" -> "repressors").
                - Choose the tool that best matches the user's intent (e.g., regulators of a gene vs. genes regulated by a TF).
                - Populate the required tool arguments (`gene_name` or `tf_name`) using the inferred entity from the conversation.

                Select the appropriate tool. Begin!""",
            ),
            # Placeholder for conversation history, managed by the agent's memory.
            MessagesPlaceholder(variable_name="chat_history"),
            # Placeholder for the user's current input.
            ("human", "{input}"),
            # Placeholder for the agent's intermediate steps (tool calls and observations).
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the Structured Chat Agent.
    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm_client.llm_gemini,
        tools=tools,
        prompt=prompt
    )

    # Create the Agent Executor, which runs the agent and its tools.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=llm_client.memory,  # Use the shared memory from the LLM client.
        verbose=True,  # Set to True to see the agent's thought process in the console.
        handle_parsing_errors=False,  # Let errors propagate for debugging.
        return_intermediate_steps=True,  # Crucial for extracting raw tool output later.
    )

    return agent_executor