# Neo4jBioRAG

A graph-based Retrieval-Augmented Generation (RAG) system using Neo4j to enhance LLM answers with structured biomedical data.

---

## Project Structure

```
Neo4jBioRAG/
│
├── README.md
├── requirements.txt
├── main.py
├── pp.py
│
├── data/
│   ├── raw/
│   │   └── net.json
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── neo4j_database.py
│   ├── dataprocessing/
│   │   ├── __init__.py
│   │   └── data_inference.py
│   ├── llm/
│   │   ├── __inint__.py
│   │   ├── llm_client.py
│   │   └── llm_client.svg
│   ├── networks/
│   │   ├── __init__.py
│   │   └── Network.py
│   ├── prompts/
│   │   ├── answer_generation_prompt.txt
│   │   └── function_call_prompt.txt
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── retriever.py
│   └── utils/
│       ├── __init__.py
│       ├── errors.py
│       ├── file_utils.py
│       └── logger_config.py
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
