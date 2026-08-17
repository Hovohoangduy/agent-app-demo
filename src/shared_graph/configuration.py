from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional, Type, TypeVar, Any
from langchain_core.runnables import RunnableConfig, ensure_config

@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.
    This class defines the parameters needed for configuring the indexing and retrieval process,
    including user identification, embedding model selection, retriever provider choice and search paramerers."""
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embedding"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["supbase", "chroma"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="chroma",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are `supbase` or `chroma`."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "desciption": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(cls: Type[T], config: Optional[RunnableConfig] = None) -> T:
        """Create an BaseConfiguration instance from a RunnableConfig object.
        
        Args:
            cls (Type[T]): The class itself.
            config (Optinal[RunnableConfig]): The configuration object to use.
        Returns:
            T: As instance of BaseConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

T = TypeVar("T", bound=BaseConfiguration)