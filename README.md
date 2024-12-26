# kblite

A SQLite-based interface for Knowledge Bases that provides efficient access to the ConceptNet knowledge base.

## Installation

```bash
pip install kblite
```

## Usage

```python
from kblite import KnowledgeBase

# Initialize from HuggingFace dataset
kb = KnowledgeBase.from_repository("ysenarath/conceptnet-sqlite")

# Query nodes
for node in kb.iternodes():
    print(node)

# Get vocabulary
vocab = kb.get_vocab()
```

## Data Files

The ConceptNet database files are hosted on HuggingFace:
https://huggingface.co/datasets/ysenarath/conceptnet-sqlite/tree/main

## License

MIT License
