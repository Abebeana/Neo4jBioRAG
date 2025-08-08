from langchain.tools import Tool
from src.database.neo4j_database import Neo4jDatabase


def get_gene_tools(database: Neo4jDatabase):
    """Returns a list of tools that use the provided database instance."""
    return [
        Tool(
            name="get_activators_of_gene",
            description="Returns genes that activate the given gene (e.g. FOXA1).",
            func=database.get_activators_of_gene,
        ),
        Tool(
            name="get_repressors_of_gene",
            description="Returns genes that repress the given gene.",
            func=database.get_repressors_of_gene,
        ),
        Tool(
            name="get_regulators_of_gene",
            description="Returns all genes that regulate (either repress or activate) the given gene.",
            func=database.get_regulators_of_gene,
        ),
        Tool(
            name="get_genes_activated_by_gene",
            description="Returns genes that are activated by the given gene.",
            func=database.get_genes_activated_by_gene,
        ),
        Tool(
            name="get_genes_repressed_by_gene",
            description="Returns genes that are repressed by the given gene.",
            func=database.get_genes_repressed_by_gene,
        ),
        Tool(
            name="get_genes_regulated_by_gene",
            description="Returns all genes that are regulated (activated or repressed) by the given gene.",
            func=database.get_genes_regulated_by_gene,
        ),
    ]
