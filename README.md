# Neo4jBioRAG

A graph-based Retrieval-Augmented Generation (RAG) system using Neo4j and LangChain to answer questions about biomedical data. This project leverages a Large Language Model (LLM) that can query a Neo4j graph database to provide accurate, context-aware answers.

---

## Features

- **Graph-Based RAG**: Uses a Neo4j database to store and query a gene regulatory network, providing structured data to the LLM.
- **LangChain Agent**: A sophisticated agent determines the user's intent, selects the appropriate database tool, and extracts the raw data.
- **Two-Step Response Generation**:
    1. The agent retrieves the raw data from the database.
    2. A separate LLM chain generates a user-friendly, natural language answer based on the retrieved data.
- **Interactive Database Loading**: Prompts the user before loading data into the database to prevent accidental overwrites.
- **Modular and Configurable**: Key components like the LLM, database, and prompts are separated into modules and configured via a `.env` file.

---

## Getting Started

Follow these steps to set up and run the project locally. For a more detailed guide, see `instructions.txt`.

### 1. Prerequisites

- Python 3.8+
- A running Neo4j database instance.

### 2. Setup

**Clone the repository:**
```bash
git clone https://github.com/Abebeana/Neo4jBioRAG.git
cd Neo4jBioRAG
```

**Create a virtual environment and install dependencies:**
```bash
python -m venv ragvenv
source ragvenv/bin/activate  # On Windows: ragvenv\Scripts\activate
pip install -r requirements.txt
```

**Configure your environment:**
Create a `.env` file in the project root and populate it with your credentials. A template is provided in `instructions.txt`. At a minimum, you will need to add your `GOOGLE_API_KEY`.

### 3. Running the Application

**Run the main script:**
```bash
python main.py
```

**First-Time Setup (Database Loading):**
The first time you run the application, you will be asked if you want to load the network data into the database. Type `yes` and press Enter.

```
Do you want to load the network data into the database? (yes/no): yes
```

This will populate your Neo4j instance with the data from `data/raw/net.json`. On subsequent runs, you can type `no` to skip this step.

---

## Project Structure

```
Neo4jBioRAG/
│
├── .env                # Stores environment variables (credentials, config)
├── .gitignore          # Specifies files to ignore for Git
├── README.md           # This file
├── requirements.txt    # Project dependencies
├── instructions.txt    # Detailed setup instructions
├── main.py             # Main application entry point
│
├── data/
│   └── raw/
│       └── net.json    # Raw gene regulatory network data
│
└── src/
    ├── agents/         # Agent creation and tool definitions
    ├── database/       # Neo4j database client and logic
    ├── llm/            # LLM client, configuration, and chains
    ├── networks/       # Network data loading utility
    ├── prompts/        # Prompt templates for the agent and LLM
