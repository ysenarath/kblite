import logging
from pathlib import Path

import plyvel
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processed_wiktdata_path = Path(
    "/Users/yasas/Downloads/Temp/2024-12-24/raw-wiktextract-data.jsonl"
)


class GraphIndex:
    def __init__(self, path: Path | str):
        path = Path(path)
        self.path = path.with_suffix(".jsonl.index")
        self.db = plyvel.DB(str(path), create_if_missing=True)
        logger.info(f"GraphIndex initialized with path: {self.path}")

    def __del__(self):
        self.db.close()
        logger.info(f"GraphIndex at path {self.path} closed")


# wiktdata_df = pd.read_json(processed_wiktdata_path, lines=True)
wikidata_lf = pl.scan_ndjson(processed_wiktdata_path, low_memory=True)

# logger.info("Schema of the wikidata_lf:")
# pprint(wikidata_lf.collect_schema())

definitions: dict[str, set[str]] = {}

head = wikidata_lf.limit(100).collect()

columns = head.collect_schema().names()

logger.info("Processing rows...")
for row in head.iter_rows():
    row = dict(zip(columns, row))
    title = row["title"] or row["word"]
    lang_code = row["lang_code"]
    lang_name = row["lang"]
    forms = row["forms"]
    senses = row["senses"]
    assert title is not None
    logger.info(f"Processing title: {title}")
    logger.info("=" * 80)
    logger.info("Title: " + title)
    if senses is None:
        continue
    for sense in senses:
        wikipedia = sense["wikipedia"]
        glosses = sense["glosses"]
        raw_glosses = sense["raw_glosses"]
        logger.info(
            f"Title: {title}, Wikipedia ID: {wikipedia}, Glosses: {glosses}, Raw Glosses: {raw_glosses}"
        )
        logger.info("-" * 80)
    logger.info("=-" * 40)

logger.info("Finished processing rows")
