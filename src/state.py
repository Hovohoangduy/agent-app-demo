import hashlib
import uuid
from typing import Annotated, Any, Literal, Optional, Union
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from dataclasses import dataclass, field

def _generate_uuid(page_content: str) -> str:
    """Generate a UUID for a document based on page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))


def _document_from_dict(item: dict[str, Any]) -> Document:
    """Create a Document from JSON using either LangChain or JS-style keys."""
    page_content = item.get("page_content", item.get("pageContent", ""))
    metadata = item.get("metadata") or {}
    item_id = metadata.get("uuid") or _generate_uuid(page_content)
    return Document(page_content=page_content, metadata={**metadata, "uuid": item_id})

def reduce_docs(
        existing: Optional[list[Document]],
        new: Union[
            list[Document],
            list[dict[str, Any]],
            list[str],
            str,
            Literal["delete"],
        ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    if new is None:
        return list(existing) if existing else []

    existing_list = list(existing) if existing else []
    existing_ids = {doc.metadata.get("uuid") for doc in existing_list}

    if isinstance(new, str):
        item_id = _generate_uuid(new)
        if item_id in existing_ids:
            return existing_list
        return existing_list + [
            Document(page_content=new, metadata={"uuid": item_id})
        ]

    if isinstance(new, Document):
        item_id = new.metadata.get("uuid") or _generate_uuid(new.page_content)
        if item_id in existing_ids:
            return existing_list

        new_item = new.copy(deep=True)
        new_item.metadata["uuid"] = item_id
        return existing_list + [new_item]

    new_list = []
    if isinstance(new, list):
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                if item_id not in existing_ids:
                    new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                    existing_ids.add(item_id)
            elif isinstance(item, dict):
                new_item = _document_from_dict(item)
                item_id = new_item.metadata["uuid"]
                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)
            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)
    return existing_list + new_list

# The index state defines the simple IO for the single-node index graph
@dataclass
class IndexState:
    """Represents the state for document indexing and retrieval.
    This class defines the structure of the index state, 
    which includes the documents to indexed and the retriever used for searching these documents."""
    docs: Annotated[list[Document], reduce_docs] = field(
        default_factory=list,
        metadata={"description": "A list of documents that the agent can index."}
    )

class AgentState(MessagesState):
    query: str
    route: str
    documents: Annotated[list[Document], reduce_docs]
