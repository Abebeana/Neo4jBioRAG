import os
import re
from unittest import result
from urllib import response
from dotenv import load_dotenv
from src import retriever
from src.llm import llm_client
from src.utils.logger_config import setup_logger
from src.database.neo4j_database import Neo4jDatabase
from src.networks.Network import Network
from src.retriever.retriever import retrieve_data
from src.llm.llm_client import LlmClient



load_dotenv()
logger = setup_logger(__name__)

def main():
      # Load network data
    raw_data = os.getenv("RAW_DATA_PATH", "data/raw/net.json")
    network = Network.load_from_json(raw_data)
    print(f"Loaded network with {len(network.GRN)} entries")
    
    database = Neo4jDatabase()
    llm_client = LlmClient(database)

    # database.store_network(network)
    # print("Network data stored in Neo4j database.")

  


    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            database.close()
            print("Exiting...")
            break
        try:
            result = retrieve_data(llm_client, query)
            print(f"Retrieved data: {result}")
            if not result:
                print("No data retrieved.")
                continue

            response = result.get('output')
            if not response:
                print("No data retrieved.")
                continue
            print(f"Response: {response}")

        except Exception as e:
            print(f"Error occurred: {e}")


            


if __name__ == "__main__":
    main()


