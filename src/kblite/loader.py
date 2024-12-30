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
    namespace: str = "https://octology.github.io/"
    version: Optional[str] = None
    verbose: bool | int = 1


class AutoKnowledgeBaseLoader(AutoModule):
    def __new__(cls, config: KnowledgeBaseLoaderConfig) -> KnowledgeBaseLoader:
        return super().__new__(cls, config)


class KnowledgeBaseLoader(BaseModule):
    config: KnowledgeBaseLoaderConfig

    def iterrows(self) -> Iterable[Dict[str, Any]]:
        """Iterate over edges."""
        raise NotImplementedError
