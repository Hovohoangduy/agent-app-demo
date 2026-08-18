import asyncio
import os
import sys

# Add src to python path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion_graph import graph

URLS = [
    "https://docs.langchain.com",
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://langchain-ai.github.io/langgraph/",
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/concepts/low_level/"
]

async def main():
    print("Loading documents from URLs...")
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    
    print(f"Loaded {len(docs)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    
    print("Ingesting chunks into Vector DB...")
    # Invoke the ingestion graph
    await graph.ainvoke({"docs": splits})
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
