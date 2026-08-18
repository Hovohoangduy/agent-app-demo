# Agent App Demo

Prototype RAG application using LangGraph, LangChain, OpenAI embeddings/chat models, and a vector store backend.

The project currently defines two LangGraph graphs:

- `indexer`: ingests document chunks into a vector store.
- `retrieval_graph`: routes a user query, retrieves relevant documents when needed, and generates an answer.

## Project Structure

```text
.
|-- main.py                    # Example LangGraph SDK client
|-- requirements.txt           # Python dependencies
|-- docker-compose.yml         # Local pgvector service
|-- .env                       # Runtime secrets, not committed
|-- data/
|   |-- rag_paper.pdf          # Source/sample PDF
|   `-- test.txt               # Long sample text about ancient Greece
|-- json/
|   |-- langgraph.json         # LangGraph graph registry
|   `-- docSplits.json         # Pre-split sample documents
`-- src/
    |-- configuration.py       # Graph configuration dataclasses
    |-- ingestion_graph.py     # Document ingestion graph
    |-- retrieval.py           # Embedding and retriever factories
    |-- retrieval_graph.py     # Query routing + RAG graph
    |-- state.py               # LangGraph state and document reducer
    `-- utils.py               # Document formatting and model loading helpers
```

## Runtime Configuration

Create a `.env` file with the required API keys and backend settings:

```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_supabase_secret_key
```

The default configuration in `src/configuration.py` uses:

- Embeddings: `openai/text-embedding-3-small`
- Retriever provider: `chroma`
- Document chunks file: `json/docSplits.json`
- Query model: `gpt-5.4-mini`

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Vector Store

The code supports two vector store paths:

- Chroma, selected by default in `IndexConfiguration.retriever_provider`.
- Supabase, selected by setting `retriever_provider` to `supabase`.

`docker-compose.yml` currently starts a local PostgreSQL/pgvector service:

```bash
docker compose up -d
```

However, the default code path uses Chroma over HTTP at `localhost:8000`. To run with the current default, start a compatible Chroma server separately or update the configuration to use Supabase/pgvector consistently.

## LangGraph Configuration

The graph registry is located at:

```text
json/langgraph.json
```

It registers:

```json
{
  "graphs": {
    "indexer": "./src/ingestion_graph.py:graph",
    "retrieval_graph": "./src/retrieval_graph.py:graph"
  }
}
```

Depending on how you run LangGraph CLI, you may need to pass this config path explicitly.

## Intended Flow

### 1. Ingest Documents

`src/ingestion_graph.py` reads documents from graph state. If no documents are passed in, it loads the default chunk file from `json/docSplits.json`.

The graph then creates a retriever and adds documents into the selected vector store.

### 2. Ask a Question

`src/retrieval_graph.py` runs this sequence:

1. `check_query_type` decides whether the query can be answered directly or needs retrieval.
2. `retrieve_documents` retrieves relevant chunks from the vector store.
3. `generate_response` formats the retrieved chunks and asks the chat model to answer.

### 3. Invoke from Client

`main.py` is an example async LangGraph SDK client. It creates a thread and streams updates from the `retrieval_graph` assistant.

## Current Notes

The core logic has been adjusted so document ingestion uses the vector store directly and retrieval uses the retriever interface. A few environment-level items are still worth checking before running end-to-end:

- The default vector store is Chroma, but `docker-compose.yml` starts pgvector instead of Chroma.
- Some import paths assume `src` is directly on `PYTHONPATH`.
- `main.py` defaults to `http://localhost:2026`; override it with `LANGGRAPH_DEPLOYMENT_URL` if your LangGraph server uses a different host or port.

## Development Notes

The local Python launcher in this environment was not available during validation, so syntax/runtime verification could not be completed here. Once Python is available, run:

```bash
python -m py_compile main.py src/configuration.py src/state.py src/retrieval.py src/ingestion_graph.py src/retrieval_graph.py src/utils.py
```

Then run the LangGraph server and test ingestion before invoking `main.py`.
