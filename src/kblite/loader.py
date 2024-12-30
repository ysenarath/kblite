from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, Optional

from nightjar import AutoModule, BaseConfig, BaseModule

__all__ = [
    "KnowledgeBaseLoaderConfig",
    "AutoKnowledgeBaseLoader",
    "KnowledgeBaseLoader",
]


class KnowledgeBaseLoaderConfig(BaseConfig, dispatch="identifier"):
    identifier: ClassVar[str]
    version: str = "0.0.1"
    namespace: str = "https://example.com/"
    verbose: bool | int = 1


class AutoKnowledgeBaseLoader(AutoModule):
    def __new__(cls, config: KnowledgeBaseLoaderConfig) -> KnowledgeBaseLoader:
        return super().__new__(cls, config)


class KnowledgeBaseLoader(BaseModule):
    config: KnowledgeBaseLoaderConfig

    def iterrows(self) -> Iterable[Dict[str, Any]]:
        """Iterate over edges."""
        raise NotImplementedError
