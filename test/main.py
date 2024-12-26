from kblite import KnowledgeBase

# Initialize from HuggingFace dataset
kb = KnowledgeBase.from_repository("ysenarath/conceptnet-sqlite")

# Query nodes
for node in kb.iternodes():
    print(node)

# Get vocabulary
vocab = kb.get_vocab()
