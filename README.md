# Neo4jBioRAG

A graph-based Retrieval-Augmented Generation (RAG) system using Neo4j to enhance LLM answers with structured biomedical data.

---

## Project Structure

```
Neo4jBioRAG/
│
├── src/
│   ├── main.py                  # Entry point for the application
│   ├── database/
│   │   └── neo4j_database.py    # Handles storage and access to Neo4j database
│   ├── dataprocessing/
│   │   └── data_inference.py    # Processes raw data and generates inferences for Neo4j
│   ├── llm/
│   │   └── llm_client.py        # Initializes and manages the LLM using prompt templates
│   ├── retriever/
│   │   └── retriever.py         # Retrieves data from Neo4j based on LLM function calls
│   ├── prompts/
│   │   ├── answer_generation_template.txt   # Guides LLM in generating user responses
│   │   └── function_call_template.txt       # Guides LLM in selecting function calls
│   └── utils/
│       └── logger_config.py     # Centralized logger configuration
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Key Modules

- **main.py**: Application entry point. Loads environment variables, sets up logging, and initializes all major modules.
- **database/neo4j_database.py**: Manages storage and retrieval of processed data in a Neo4j graph database.
- **dataprocessing/data_inference.py**: Processes raw biomedical data, generates inferences, and prepares data for Neo4j.
- **llm/llm_client.py**: Initializes the Large Language Model (LLM) using environment variables and prompt templates. Guides the LLM in function selection and response generation.
- **retriever/retriever.py**: Retrieves relevant data from Neo4j in response to LLM-generated function calls, feeding results back to the LLM for answer generation.
- **prompts/answer_generation_template.txt**: Template for guiding the LLM in generating user responses using retrieved data.
- **prompts/function_call_template.txt**: Template for instructing the LLM on how to select and format function calls for data retrieval.
- **utils/logger_config.py**: Provides a centralized logger setup.

---
