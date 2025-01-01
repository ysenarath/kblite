from typing import Generator, Set, Tuple

from kblite import KnowledgeBase, KnowledgeBaseConfig
from kblite.flashtext import KeywordProcessor

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
    kp.add_keyword(key)


def analyze_text(text: str) -> Generator[Tuple[str, int, int, Set[str]], None, None]:
    # text = "/r/Shit_Chapo_Says is controlled opposition. They ban conservatives and republicans but let Chapos shitpost in the sub with no repercussion. [linebreak]  [linebreak] It's run by a bunch of neolibs and socialists who side more with CTH than centrists but just don't like the flavor of socialism that CTH likes."
    keywords = kp.extract_keywords(text, span_info=True, max_cost=2)
    for term, start, end in keywords:
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
