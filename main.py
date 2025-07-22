import os
from dotenv import load_dotenv
from src import retriever
from src.llm import llm_client
from src.utils.logger_config import setup_logger
from src.database.neo4j_database import Neo4jDatabase
from src.networks.Network import Network
from src.retriever.retriever import Retriever
from src.llm.llm_client import LlmClient



load_dotenv()
logger = setup_logger(__name__)

def main():
      # Load network data
    raw_data = os.getenv("RAW_DATA_PATH", "data/raw/net.json")
    network = Network.load_from_json(raw_data)
    print(f"Loaded network with {len(network.GRN)} entries")
    
    llm_client = LlmClient()         
    database = Neo4jDatabase()       
    retriever = Retriever(llm_client, database) 


    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            database.close()
            print("Exiting...")
            break

        # Use the LLM client to process the query
        retrieved_data = retriever.retrieve_data(query)
        if retrieved_data:
            print(f"Retrieved data: {retrieved_data}")
        else:
            print("No data retrieved.")
            continue
        llm_response = llm_client.llm_biomistral.invoke(query)

        print(f"LLM response: {llm_response}")


if __name__ == "__main__":
    main()


