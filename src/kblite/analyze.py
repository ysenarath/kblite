from typing import Generator, List, Set, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from kblite import KnowledgeBase, KnowledgeBaseConfig
from kblite.core import Embedding
from kblite.flashtext import KeywordProcessor
from kblite.lexrank import degree_centrality_scores

# Initialize stopwords
STOPWORDS = set(stopwords.words("english"))

config = KnowledgeBaseConfig.from_dict(
    {
        "loader": {
            "identifier": "conceptnet",
        },
    }
)

kb = KnowledgeBase(config)
kp = KeywordProcessor()

eb = Embedding(
    {
        "identifier": "fasttext",
    }
)

for key in set(kb.vocab.keys()):
    if key.lower() in STOPWORDS or len(key) < 3:
        # do not index stopwords or short words
        continue
    kp.add_keyword(key)


def analyze_text(
    text: str, max_cost: int = 0
) -> Generator[Tuple[str, int, int, Set[str]], None, None]:
    # max_cost = 2 => fuzzy matching
    keywords: List[Tuple[str, int, int]] = kp.extract_keywords(
        text, span_info=True, max_cost=max_cost
    )
    for term, start, end in keywords:
        # Double check for stopwords (in case of case sensitivity)
        try:
            # find all forms of the term (if any)
            forms = list(zip(*kb.triplets.find(term, rel="FormOf")))
            if len(forms) == 3:
                forms = forms[2]
        except ValueError:
            forms = []
        forms = set(forms)
        forms.add(term)
        triples = set()
        for form in forms:
            for rel in ["HasContext", "IsA"]:
                triples.update(kb.triplets.find(form, rel=rel))
        for c in triples:
            yield (term, start, end, c)


def get_scores(
    triples: List[Tuple[str, int, int, Set[str]]],
    nlargest: int = 2,
) -> List[Tuple[str, int]]:
    vectors = {}
    vocab = set()
    for term, _, _, (s, v, o) in triples:
        vocab.update([term, s, o])
    vocab = list(vocab)
    for term in vocab:
        vectors[term] = eb.get_word_vector(term)
    similarity_matrix = np.zeros((len(vocab), len(vocab)))
    for i, term in enumerate(vocab):
        for j, other_term in enumerate(vocab):
            if term == other_term:
                continue
            a, b = vectors[term], vectors[other_term]
            s = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            similarity_matrix[i, j] = s
    try:
        scores = degree_centrality_scores(similarity_matrix, threshold=0.1)
    except ValueError:
        scores = np.zeros(len(vocab))
    scores = dict(zip(vocab, scores))
    out = []
    for term, start, end, (s, v, o) in triples:
        avg_score = (scores[s] + scores[o]) / 2
        out += [(term, start, end, (s, v, o), avg_score)]
    # select top 2 per term
    out = (
        pd.DataFrame(out, columns=["term", "start", "end", "triple", "score"])
        .groupby("term")
        .apply(lambda x: x.nlargest(nlargest, "score"))
        .reset_index(drop=True)
    )
    return [row for row in out.itertuples(index=False)]
