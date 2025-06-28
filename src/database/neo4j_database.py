""" This module handles the storage of processed data in a Neo4j database, 
enabling efficient data accessibility and management."""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.logger_config import setup_logger
from neo4j import GraphDatabase
from src.networks.Network import Network
import pandas as pd

load_dotenv()
logger = setup_logger(__name__)




class Neo4jDatabase:
    """A class to handle interactions with a Neo4j database for storing and retrieving processed data """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """Initialize the Neo4jDatabase with connection parameters."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Neo4jDatabase initialized.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self.driver = None

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4jDatabase connection closed.")

    def store_network(self, network: Network):
        """Store a Network object in the Neo4j database."""
        if not self.driver:
            logger.error("Database connection is not established.")
            return

        try:
            self._populate_database(network)
            logger.info("Network data successfully stored in Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to store network data: {e}")
    
    def _create_constraints(self):
        """Create constraints for the nodes in the Neo4j database."""
        if not self.driver:
            logger.error("Database connection is not established.")
            return

        with self.driver.session() as session:
            try:
                # Create unique constraints for the main node types
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tf:TranscriptionFactor) REQUIRE tf.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE")
                logger.info("Constraints created successfully.")
            except Exception as e:
                logger.error(f"Constraint creation failed: {e}")

    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """Execute a Cypher query on the Neo4j database."""
        if not self.driver:
            logger.error("Database connection is not established.")
            return None
        try:

            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return result.data()
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return None
        



    def _populate_database(self, network: Network):
        """Populate the Neo4j database with the network data using intermediate Interaction nodes to represent regulatory contexts."""
        if not self.driver:
            logger.error("Database connection is not established.")
            return

        self._create_constraints()
        grn_df = pd.DataFrame(network.GRN)
        

        unique_targets = set(grn_df.index)

        def flatten(list_of_lists):
            flat_list = []
            for sublist in list_of_lists:
                if isinstance(sublist, list):
                    for item in sublist:
                        flat_list.append(item)
            return flat_list
        def is_tf_present(coact_value, tf):
            """Check if a transcription factor is present in the coact or corep lists."""
            if isinstance(coact_value, list):
                return tf in coact_value
            return coact_value == tf


        coacts = flatten(grn_df["coact"].tolist())
        coreps = flatten(grn_df["corep"].tolist())
        unique_tfs = set(coacts + coreps)

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
                subset = grn_df.loc[[target]] 
                filtered = subset[subset["coact"].apply(is_tf_present, tf=tf)]
                
                for idx, row in filtered.iterrows():
                    interaction_id += 1
                    all_coacts = row.get("coact", []).remove(tf)
                    if all_coacts:
                        coactivators_str = "_" + "_".join(all_coacts)
                        context = f"{tf}_with_other_coactivators{coactivators_str}"
                    else:
                        context = f"{tf}_with_no_coactivators"
                                                         
                        # Create Interaction node properties
                    interaction_props = {
                        "id": f"interaction_{interaction_id}",
                        "context": context,
                        "regulation_type": "ACTIVATES",
                        "coef_acts": row.get("Coef_Acts", None),
                        "coef_coacts": row.get("Coef_coActs", None),
                        "r2": row.get("R2", None),
                        "rmse": row.get("RMSE", None)
                    }
                   
                    
                    # Create the Interaction node
                    self.execute_query(
                        """
                        MERGE (i:Interaction {id: $id})
                        SET i += $props
                        """,
                        {"id": interaction_props["id"], "props": interaction_props}
                    )
                    
                    # Create PARTICIPATES_IN relationship from TF to Interaction
                    self.execute_query(
                        """
                        MATCH (tf:TranscriptionFactor {name: $tf}), (i:Interaction {id: $interaction_id})
                        MERGE (tf)-[:PARTICIPATES_IN]->(i)
                        """,
                        {"tf": tf, "interaction_id": interaction_props["id"]}
                    )
                    
                    # Create REGULATES relationship from Interaction to Gene
                    self.execute_query(
                        """
                        MATCH (i:Interaction {id: $interaction_id}), (g:Gene {name: $target})
                        MERGE (i)-[:REGULATES]->(g)
                        """,
                        {"interaction_id": interaction_props["id"], "target": target}
                    )
            
            # For repressors
            for tf in regulators.get('rep', []):
                subset = grn_df.loc[[target]] 
                filtered = subset[subset["corep"].apply(is_tf_present, tf=tf)]
                
                for idx, row in filtered.iterrows():
                    interaction_id += 1
                    all_coreps = row.get("corep", []).remove(tf)
                    if all_coreps:
                        corepressors_str = "_" + "_".join(all_coreps)
                        context = f"{tf}_with_other_corepressors{corepressors_str}"
                    else:
                        context = f"{tf}_with_no_corepressors"
                    
                    # Create Interaction node properties
                    interaction_props = {
                        "id": f"interaction_{interaction_id}",
                        "context": context,
                        "regulation_type": "REPRESSES",
                        "coef_reps": row.get("Coef_Reps", None),
                        "coef_coreps": row.get("Coef_coReps", None),
                        "r2": row.get("R2", None),
                        "rmse": row.get("RMSE", None)
                    }
                    # Remove None values
                    interaction_props = {k: v for k, v in interaction_props.items() if v is not None}
                    
                    # Create the Interaction node
                    self.execute_query(
                        """
                        MERGE (i:Interaction {id: $id})
                        SET i += $props
                        """,
                        {"id": interaction_props["id"], "props": interaction_props}
                    )
                    
                    # Create PARTICIPATES_IN relationship from TF to Interaction
                    self.execute_query(
                        """
                        MATCH (tf:TranscriptionFactor {name: $tf}), (i:Interaction {id: $interaction_id})
                        MERGE (tf)-[:PARTICIPATES_IN]->(i)
                        """,
                        {"tf": tf, "interaction_id": interaction_props["id"]}
                    )
                    
                    # Create REGULATES relationship from Interaction to Gene
                    self.execute_query(
                        """
                        MATCH (i:Interaction {id: $interaction_id}), (g:Gene {name: $target})
                        MERGE (i)-[:REGULATES]->(g)
                        """,
                        {"interaction_id": interaction_props["id"], "target": target}
                    )
