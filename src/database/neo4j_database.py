""" This module handles the storage of processed data in a Neo4j database, 
enabling efficient data accessibility and management."""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.logger_config import setup_logger
from neo4j import GraphDatabase
from src.networks.Network import Network
import pandas as pd
from functools import wraps
from src.utils.flattener import flatten_series as flatten

load_dotenv()
logger = setup_logger(__name__)




# Decorator for session management
def with_session(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'driver') or self.driver is None:
            raise RuntimeError("Database driver is not initialized.")
        with self.driver.session() as session:
            return func(self, *args, session=session, **kwargs)
    return wrapper

def method_cache(func): 
    cache = {}
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        key = (func.__name__, args, frozenset(kwargs.items())) 
          # only immutable objects can be used as keys in a dictionary
        if key in cache:
            return cache[key]
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result

    return wrapper

class Neo4jDatabase:
    """A class to handle interactions with a Neo4j database for storing and retrieving processed data """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize the Neo4jDatabase with connection parameters."""
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            logger.info("Neo4jDatabase initialized.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self.driver = None


    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4jDatabase connection closed.")

    @with_session
    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, *, session):
        try:
            result = session.run(query, parameters or {})
            return result.data()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):  
        # The session parameter is intentionally omitted here because:
        # 1. It's a keyword-only argument (enforced by *, in _execute_query signature)
        # 2. The @with_session decorator will inject it at runtime
        # 3. The parameters argument being Optional has no effect on session being required
        # 4. Type checker warning is suppressed (# type: ignore) because we guarantee injection
        return self._execute_query(query, parameters) # type: ignore


    def _create_constraints(self):
        try:
            self.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE")
            self.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (tf:TranscriptionFactor) REQUIRE tf.name IS UNIQUE")
            # self.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE")
            logger.info("Constraints created.")
        except Exception as e:
            logger.error(f"Constraint creation failed: {e}")




   
    def store_network(self, network: Network):
        try:
            self._populate_database(network)
            logger.info("Network data stored in Neo4j.")
        except Exception as e:
            logger.error(f"Failed to store network: {e}")
            


    def _populate_database(self, network: Network):
        """Populate the Neo4j database with the network data using intermediate Interaction nodes to represent regulatory contexts."""
      
        self._create_constraints()
        grn_df = pd.DataFrame(network.GRN)
        
        unique_targets = set(grn_df.index)
        coacts = set(flatten(grn_df["coact"]))
        coreps = set(flatten(grn_df["corep"]))
        unique_tfs = coreps.union(coacts)

        # Create all Gene and TranscriptionFactor nodes first
        for target in unique_targets:
            self.execute_query(
                "MERGE (g:Gene {name: $name})",
                {"name": target}
            )
        for tf in unique_tfs:
            self.execute_query(
                "MERGE (tf:TranscriptionFactor {name: $name})",
                {"name": tf}
            )

        # Create Interaction nodes and relationships for each regulatory context
        interaction_id = 0
        for target in unique_targets:
            regulators = network.get_regulators(target)
            
            # For activators
            for tf in regulators.get('act', []):

                self.execute_query("""
                    MATCH (g:Gene {name: $target}), (tf:TranscriptionFactor {name: $tf})
                    MERGE (tf)-[:ACTIVATES]->(g)
                """, {"target": target, "tf": tf})

            # For repressors
            for tf in regulators.get('rep', []):
                self.execute_query("""
                    MATCH (g:Gene {name: $target}), (tf:TranscriptionFactor {name: $tf})
                    MERGE (tf)-[:REPRESSES]->(g)
                """, {"target": target, "tf": tf})


    @method_cache
    def get_activators_of_gene(self, gene_name: str) -> List[Dict]:
        """Return all transcription factors that activate the given gene."""
        query = (
            "MATCH (tf:TranscriptionFactor)-[:ACTIVATES]->(g:Gene {name: $gene_name}) "
            "RETURN tf.name AS activator"
        )
        return self.execute_query(query, {"gene_name": gene_name})

    @method_cache
    def get_repressors_of_gene(self, gene_name: str) -> List[Dict]:
        """Return all transcription factors that repress the given gene."""
        query = (
            "MATCH (tf:TranscriptionFactor)-[:REPRESSES]->(g:Gene {name: $gene_name}) "
            "RETURN tf.name AS repressor"
        )
        return self.execute_query(query, {"gene_name": gene_name})

    @method_cache
    def get_regulators_of_gene(self, gene_name: str) -> List[Dict]:
        """Return all transcription factors that regulate (activate or repress) the given gene."""
        query = (
            "MATCH (tf:TranscriptionFactor)-[r]->(g:Gene {name: $gene_name}) "
            "WHERE type(r) IN ['ACTIVATES', 'REPRESSES'] "
            "RETURN tf.name AS regulator, type(r) AS regulation_type"
        )
        return self.execute_query(query, {"gene_name": gene_name})

    @method_cache
    def get_genes_repressed_by_gene(self, tf_name: str) -> List[Dict]:
        """Return all genes repressed by the given transcription factor."""
        query = (
            "MATCH (tf:TranscriptionFactor {name: $tf_name})-[:REPRESSES]->(g:Gene) "
            "RETURN g.name AS repressed_gene"
        )
        return self.execute_query(query, {"tf_name": tf_name})

    @method_cache
    def get_genes_activated_by_gene(self, tf_name: str) -> List[Dict]:
        """Return all genes activated by the given transcription factor."""
        query = (
            "MATCH (tf:TranscriptionFactor {name: $tf_name})-[:ACTIVATES]->(g:Gene) "
            "RETURN g.name AS activated_gene"
        )
        return self.execute_query(query, {"tf_name": tf_name})

    @method_cache
    def get_genes_regulated_by_gene(self, tf_name: str) -> List[Dict]:
        """Return all genes regulated (activated or repressed) by the given transcription factor."""
        query = (
            "MATCH (tf:TranscriptionFactor {name: $tf_name})-[r]->(g:Gene) "
            "WHERE type(r) IN ['ACTIVATES', 'REPRESSES'] "
            "RETURN g.name AS gene, type(r) AS regulation_type"
        )
        return self.execute_query(query, {"tf_name": tf_name})

    def dumb_method_one(self):
        """A placeholder method that does nothing."""
        return None

    def dumb_method_two(self):
        """Another placeholder method that does nothing."""
        return None

        

