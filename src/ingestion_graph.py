import inspect
import json
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from configuration import IndexConfiguration
from state import IndexState, reduce_docs
from retrieval import make_vectorstore


async def _add_documents(vectorstore, docs) -> None:
    async_add = getattr(vectorstore, "aadd_documents", None)
    if async_add is not None:
        try:
            result = async_add(docs)
            if inspect.isawaitable(result):
                await result
            return
        except NotImplementedError:
            pass

    add = getattr(vectorstore, "add_documents", None)
    if add is None:
        raise AttributeError("Vector store does not support adding documents.")

    add(docs)

async def ingest_docs(state: IndexState, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    configuration = IndexConfiguration.from_runnable_config(config)
    docs = state.docs if isinstance(state, IndexState) else state["docs"]
    if not docs:
        with open(configuration.docs_file, encoding="utf-8") as file_content:
            serialized_docs = json.loads(file_content.read())
            docs = reduce_docs([], serialized_docs)
    else:
        docs = reduce_docs([], docs)

    with make_vectorstore(configuration) as vectorstore:
        await _add_documents(vectorstore, docs)

    return {"docs": "delete"}

# define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(ingest_docs)
builder.add_edge(START, "ingest_docs")
builder.add_edge("ingest_docs", END)

# compile into a graph object that you can invoke and deploy
graph = builder.compile()
graph.name = "IngestionGraph"
