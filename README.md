# Agent App Demo (FastAPI + LangGraph RAG Chatbot)

This is a Retrieval-Augmented Generation (RAG) prototype application built using **LangGraph**, **LangChain**, and OpenAI models. The application provides a modern web-based Chatbot interface using **FastAPI** as the backend and a **PostgreSQL (pgvector)** database for vector storage.

The chatbot features:
- **Streaming**: Real-time typing effect responses (Server-Sent Events).
- **Smart Routing**: Differentiates between general conversational questions (direct answers) and knowledge retrieval queries (executing the RAG pipeline).
- **Modern UI**: Features an elegant Dark Mode, automatic Markdown rendering (bold text, lists, code blocks), and smooth micro-animations.

## Project Structure

```text
.
|-- main.py                    # Main Application (FastAPI Server)
|-- requirements.txt           # Python dependencies
|-- pyproject.toml             # Project configuration and dependencies
|-- docker-compose.yml         # Configuration to run PostgreSQL (pgvector) via Docker
|-- .env                       # Environment variables (API Keys, Configs)
|
|-- scripts/
|   `-- ingest_docs.py         # Script to automatically crawl web documentation and ingest into the DB
|
|-- static/                    # Web Interface (Frontend)
|   |-- index.html             # HTML structure of the chat page
|   |-- style.css              # Styling, colors, animations (Dark mode)
|   `-- script.js              # API communication logic, streaming, and markdown rendering
|
|-- json/
|   |-- langgraph.json         # (Deprecated) Old LangGraph registry
|   `-- docSplits.json         # Sample static chunk data
|
`-- src/                       # Core LangGraph Agent logic
    |-- configuration.py       # Configuration settings (model, retriever, default provider is pgvector)
    |-- ingestion_graph.py     # Graph logic for document ingestion into the Vector DB
    |-- retrieval.py           # VectorStore initialization (supports pgvector, chroma, supabase)
    |-- retrieval_graph.py     # Graph for Smart Routing and RAG
    |-- state.py               # Agent state management
    `-- utils.py               # Utility functions for document formatting and model initialization
```

## Usage Instructions

### 1. Prerequisites
- **Python 3.9+** installed.
- **Docker & Docker Compose** installed (to run the database).
- An OpenAI API Key.

### 2. Environment Variables Setup
Create a `.env` file in the root directory of the project and add the following information:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies
Install the required dependencies via pip within a virtual environment (e.g., conda env):
```bash
pip install -r requirements.txt
```

### 4. Start the Database (PostgreSQL/pgvector)
The system uses pgvector for document storage. Start the database using:
```bash
docker compose up -d
```
*Note: The database will run on port `6024` as configured by default in the docker-compose.yml file.*

### 5. Data Ingestion
Before running the chatbot, populate it with knowledge. Run the following script to automatically fetch documentation from web pages (LangChain, LangGraph docs) and load them into the database:
```bash
python scripts/ingest_docs.py
```
*(Please wait a moment for the script to fetch, chunk the text, and save all the data into the DB).*

### 6. Start the Web Server (FastAPI)
Launch the backend Chatbot application using `uvicorn`:
```bash
uvicorn main:app --reload --port 8000
```

### 7. Experience the Chatbot
- Open your web browser and navigate to: **http://localhost:8000**
- A modern chat interface will appear. You can try asking questions like *"What is LangGraph?"* or *"How does State work in LangGraph?"* to witness the real-time response speed, clearly formatted Markdown UI, and the AI assistant's information retrieval capabilities.
