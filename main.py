import os
from dotenv import load_dotenv
from src.utils.logger_config import setup_logger
from src.database.neo4j_database import Neo4jDatabase
from src.networks.Network import Network

load_dotenv()
logger = setup_logger(__name__)

if __name__ == "__main__":
  
    
    # Load network data
    raw_data = os.getenv("RAW_DATA_PATH", "data/raw/net.json")
    network = Network.load_from_json(raw_data)
    print(f"Loaded network with {len(network.GRN)} entries")
    
    # Test database
    db = Neo4jDatabase()
    if db.driver:
        print("Connected to Neo4j")
        db.store_network(network)
        print("Network stored in database")
        stats = network.metadata
        
        print(f"Network metadata: {stats}")
        
        db.close()
        print("Done!")
    else:
        print(" Failed to connect to Neo4j")
