import inspect
from typing import Literal
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from utils import format_docs, load_chat_model
from configuration import Configuration
from retrieval import make_retriever
from state import AgentState

class Schema(BaseModel):
    route: Literal["retrieve", "direct"]

async def check_query_type(state: AgentState, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    structured_llm = load_chat_model(configuration.query_model).with_structured_output(Schema)
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a routing assistant. Your job is to determine if a question needs document retrieval or can be answered directly.\n\nRespond with 'retrieve' if the question requires retrieving documents, or 'direct' if it can be answered from general knowledge."),
        ("human", "{query}")
    ])

    formatted_prompt = routing_prompt.invoke({"query": state["query"]})
    response = structured_llm.invoke(formatted_prompt)

    route = response.route

    if route == "retrieve":
        return {"route": "retrieve_documents"}
    else:
        return {"route": "generate_direct_answer"}

async def route_query(state: AgentState, *, config: RunnableConfig):
    route = state["route"]
    if not route:
        raise ValueError("Route is not set")
    if route == "retrieve_documents":
        return "retrieve_documents"
    else:
        return "generate_direct_answer"


async def retrieve_documents(state: AgentState, *, config: RunnableConfig):
    with make_retriever(config) as retriever:
        if hasattr(retriever, "ainvoke"):
            response = retriever.ainvoke(state["query"])
            if inspect.isawaitable(response):
                response = await response
        else:
            response = retriever.invoke(state["query"])
    return {"documents": response}

async def generate_response(state: AgentState, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    context = format_docs(state["documents"])
    prompt_template = hub.pull("rlm/rag-prompt")
    formatted_prompt = prompt_template.invoke({"context": context, "question": state["query"]})
    messages = formatted_prompt.messages + state["messages"]
    model = load_chat_model(configuration.query_model).with_config(tags=["response_model"])
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def generate_direct_answer(state: AgentState, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_config(tags=["response_model"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Please answer the user's question directly."),
        ("human", "{query}")
    ])
    formatted_prompt = prompt.invoke({"query": state["query"]})
    response = await model.ainvoke(formatted_prompt)
    return {"messages": [response]}

builder = StateGraph(AgentState, config_schema=Configuration)
builder.add_node("check_query_type", check_query_type)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("generate_response", generate_response)
builder.add_node("generate_direct_answer", generate_direct_answer)
builder.add_edge(START, "check_query_type")
builder.add_conditional_edges("check_query_type", route_query)
builder.add_edge("retrieve_documents", "generate_response")
builder.add_edge("generate_response", END)
builder.add_edge("generate_direct_answer", END)

# compile
graph = builder.compile()
graph.name = "RetrievalGraph"
