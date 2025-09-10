from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from src.database.neo4j_database import Neo4jDatabase


class GeneNameInput(BaseModel):
    gene_name: str = Field(..., description="The target gene symbol.")


class TFNameInput(BaseModel):
    tf_name: str = Field(..., description="The transcription factor symbol.")


def get_gene_tools(database: Neo4jDatabase):
    """Returns a list of structured tools that use the provided database instance."""
    return [
        StructuredTool.from_function(
            name="get_activators_of_gene",
            description="Returns transcription factors that activate the given gene.",
            func=database.get_activators_of_gene_data,
            args_schema=GeneNameInput,
        ),
        StructuredTool.from_function(
            name="get_repressors_of_gene",
            description="Returns transcription factors that repress the given gene.",
            func=database.get_repressors_of_gene_data,
            args_schema=GeneNameInput,
        ),
        StructuredTool.from_function(
            name="get_regulators_of_gene",
            description="Returns all transcription factors that regulate (either repress or activate) the given gene.",
            func=database.get_regulators_of_gene_data,
            args_schema=GeneNameInput,
        ),
        StructuredTool.from_function(
            name="get_genes_activated_by_gene",
            description="Returns genes that are activated by the given transcription factor.",
            func=database.get_genes_activated_by_gene_data,
            args_schema=TFNameInput,
        ),
        StructuredTool.from_function(
            name="get_genes_repressed_by_gene",
            description="Returns genes that are repressed by the given transcription factor.",
            func=database.get_genes_repressed_by_gene_data,
            args_schema=TFNameInput,
        ),
        StructuredTool.from_function(
            name="get_genes_regulated_by_gene",
            description="Returns all genes that are regulated (activated or repressed) by the given transcription factor.",
            func=database.get_genes_regulated_by_gene_data,
            args_schema=TFNameInput,
        ),
    ]
