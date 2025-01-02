from typing import Generator, List, Set, Tuple

from nltk.corpus import stopwords

from kblite import KnowledgeBase, KnowledgeBaseConfig
from kblite.flashtext import KeywordProcessor

# Initialize stopwords
STOP_WORDS = set(stopwords.words("english"))

config = KnowledgeBaseConfig.from_dict(
    {
        "loader": {
            "identifier": "conceptnet",
        },
    }
)

kb = KnowledgeBase(config)
kp = KeywordProcessor()

for key in set(kb.vocab.keys()):
    if key.lower() in STOP_WORDS or len(key) < 3:
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
