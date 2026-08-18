import sys
import os
import uuid

# Add src to python path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from retrieval_graph import graph

app = FastAPI(title="Agent App Demo")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

from fastapi.responses import FileResponse, StreamingResponse
import json

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    input_data = {
        "query": request.query,
        "messages": []
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    async def event_generator():
        try:
            # version="v2" is required for astream_events in langgraph
            async for event in graph.astream_events(input_data, config, version="v2"):
                kind = event["event"]
                tags = event.get("tags", [])
                
                # We specifically look for tokens streamed from the chat model
                if kind == "on_chat_model_stream" and "response_model" in tags:
                    content = event["data"]["chunk"].content
                    if isinstance(content, str) and content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                        
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Error during graph execution: {e}")
            yield f"data: {json.dumps({'error': 'Sorry, an internal error occurred.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
