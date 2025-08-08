from langchain.agents import AgentExecutor, StructuredChatAgent
from src.agents.gene_tools import get_gene_tools
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from src.database.neo4j_database import Neo4jDatabase
from src.llm.llm_client import LlmClient


def create_agent(
   llm_client:LlmClient,
   database: Neo4jDatabase,
) -> AgentExecutor:
    """
    Creates and configures a structured chat agent.
    """
    tools = get_gene_tools(database)
    prompt = StructuredChatAgent.create_prompt(
        tools=tools,
        prefix="""You are a helpful assistant that answers questions about gene regulatory networks.
Use the provided tools to answer the user's questions.
If the user's input is not a question that can be answered by the tools, or if you don't know the answer, respond conversationally.
""",
        suffix="Begin!\n\n{chat_history}\n\nInput: {input}\n{agent_scratchpad}",
        input_variables=["input", "agent_scratchpad", "chat_history"],
    )

    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm_client.llm_gemini, tools=tools, prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=llm_client.memory,
        verbose=False,
        handle_parsing_errors=True,
    )

    return agent_executor