from typing import Generator, Set, Tuple

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

# Only add non-stopword keywords
for key in set(kb.vocab.keys()):
    if key.lower() not in STOP_WORDS:
        kp.add_keyword(key)


def analyze_text(text: str) -> Generator[Tuple[str, int, int, Set[str]], None, None]:
    keywords = kp.extract_keywords(text, span_info=True, max_cost=2)
    for term, start, end in keywords:
        # Double check for stopwords (in case of case sensitivity)
        if term.lower() in STOP_WORDS or len(term) < 3:
            continue
        try:
            forms = list(zip(*kb.triplets.find(term, rel="FormOf")))
            if len(forms) == 3:
                forms = forms[2]
        except ValueError:
            forms = []
        forms = set(forms)
        forms.add(term)
        contexts = set()
        for form in forms:
            for rel in ["HasContext", "IsA"]:
                contexts.update(kb.triplets.find(form, rel=rel))
        for c in contexts:
            yield (term, start, end, c)
