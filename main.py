import asyncio
import os
from langgraph_sdk import get_client

async def invoke_retrieval_assistant():
    # Initialize the Langgraph client
    deployment_url = os.getenv("LANGGRAPH_DEPLOYMENT_URL", "http://localhost:2026")
    client = get_client(url=deployment_url)

    try:
        # create new thread
        thread = await client.threads.create(
            metadata={
                "user_id": "example_user",
                "session": "retrieval_session"
            }
        )

        # Prepare the input for the retrieval graph
        input_data = {
            "query": "What is this document about?",
        }

        # Invoke the assistant on the created thread
        thread_id = thread["thread_id"] if isinstance(thread, dict) else thread.thread_id
        async for event in client.runs.stream(
            thread_id=thread_id,
            assistant_id="retrieval_graph",
            input=input_data,
            stream_mode="updates" # streams updates as they occur
        ):
            print(f"Receiving event of type: {event.event}")
            print(event.data)
            print("\n")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(invoke_retrieval_assistant())
