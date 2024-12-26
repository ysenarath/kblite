from .base import KnowledgeBase
from .conceptnet import ConceptNetLoader
from .loader import KnowledgeLoader

__all__ = [
    "KnowledgeBase",
    "KnowledgeLoader",
]

# add loaders here so the import is not removed by isort
_ConceptNetLoader = ConceptNetLoader
