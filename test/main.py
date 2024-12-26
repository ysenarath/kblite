from kblite.base import KnowledgeBase

kb = KnowledgeBase.from_repository("ysenarath/conceptnet-sqlite")

print(f"Number of edges: {kb.num_edges()}")

vocab = kb.get_vocab()

for item in vocab:
    print(item)
    break


print(kb.index.find_by_object("apple"))
