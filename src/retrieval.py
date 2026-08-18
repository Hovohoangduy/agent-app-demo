from contextlib import contextmanager
import os
from typing import Union
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
import chromadb
from dotenv import load_dotenv

from configuration import BaseConfiguration, IndexConfiguration

load_dotenv(dotenv_path=".env")
ConfigurationLike = Union[RunnableConfig, BaseConfiguration, IndexConfiguration]

def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    if "/" not in model:
        raise ValueError(
            "Embedding model must be specified as 'provider/model-name'."
        )

    provider, model_name = model.split("/", maxsplit=1)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _resolve_configuration(config: ConfigurationLike) -> BaseConfiguration:
    if isinstance(config, BaseConfiguration):
        return config
    if isinstance(config, IndexConfiguration):
        return config
    return IndexConfiguration.from_runnable_config(config)


@contextmanager
def make_supabase_vectorstore(configuration: BaseConfiguration, embedding_model: Embeddings):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_SECRET_KEY env variables")

    client = create_client(supabase_url, supabase_key)
    vectorstore = SupabaseVectorStore(
        client=client, embedding=embedding_model, table_name="documents", query_name="match_documents"
    )
    yield vectorstore

@contextmanager
def make_chroma_vectorstore(configuration: BaseConfiguration, embedding_model: Embeddings):
    client = chromadb.HttpClient(host='localhost', port=8000)

    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embedding_model,
        client=client
    )
    yield vectorstore


@contextmanager
def make_pgvector_vectorstore(configuration: BaseConfiguration, embedding_model: Embeddings):
    connection_string = "postgresql+psycopg2://langchain:langchain@localhost:6024/langchain"
    from langchain_community.vectorstores import PGVector
    
    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embedding_model,
        collection_name="documents"
    )
    yield vectorstore


@contextmanager
def make_vectorstore(config: ConfigurationLike):
    """Create a vector store based on the current configuration."""
    configuration = _resolve_configuration(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    if configuration.retriever_provider == "supabase":
        with make_supabase_vectorstore(configuration, embedding_model) as vectorstore:
            yield vectorstore
    elif configuration.retriever_provider == "chroma":
        with make_chroma_vectorstore(configuration, embedding_model) as vectorstore:
            yield vectorstore
    elif configuration.retriever_provider == "pgvector":
        with make_pgvector_vectorstore(configuration, embedding_model) as vectorstore:
            yield vectorstore
    else:
        raise ValueError(
            "Unrecognized retriever_provider in configuration."
        )


@contextmanager
def make_retriever(config: ConfigurationLike):
    """Create a retriever for the agent based on the current configuration."""
    configuration = _resolve_configuration(config)
    search_kwargs = dict(configuration.search_kwargs)
    with make_vectorstore(configuration) as vectorstore:
        yield vectorstore.as_retriever(search_kwargs=search_kwargs)
