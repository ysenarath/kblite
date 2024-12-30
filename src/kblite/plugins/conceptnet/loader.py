from __future__ import annotations

import gzip
import json
import logging
from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, Iterable, Optional, Tuple

import orjson
import requests
from tqdm import auto as tqdm

from kblite.config import config as kblite_config
from kblite.loader import KnowledgeBaseLoader, KnowledgeBaseLoaderConfig
from kblite.plugins.conceptnet import csv

__all__ = [
    "ConceptNetLoaderConfig",
    "ConceptNetLoader",
]

logger = logging.getLogger(__name__)


def get_download_urls() -> Dict[str, str]:
    base = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges"
    return {
        "5.7.0": f"{base}/conceptnet-assertions-5.7.0.csv.gz",
    }


class ConceptNetLoaderConfig(KnowledgeBaseLoaderConfig):
    identifier: ClassVar[str] = "conceptnet"
    version: str = "5.7.0"
    namespace: str = "http://conceptnet.io/"
    download_urls: Dict[str, str] = field(default_factory=get_download_urls)
    force: bool = False


def write_to_file(
    filepath: str | Path,
    generator: Generator[str, None, None],
    buffer_size: int = 1000,
):
    buffer = []
    if str(filepath).endswith(".gz"):
        open_ = gzip.open(filepath, "at")
    else:
        open_ = open(filepath, "a")
    with open_ as f:
        for obj in generator:
            if not isinstance(obj, str):
                obj = json.dumps(obj, ensure_ascii=False).replace("\\u0000", "")
            if not obj.endswith("\n"):
                obj += "\n"
            buffer.append(obj)
            if len(buffer) >= buffer_size:
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer)


class ConceptNetLoader(KnowledgeBaseLoader):
    config: ConceptNetLoaderConfig

    def __init__(self, config: Optional[ConceptNetLoaderConfig] = None) -> None:
        if config is None:
            config = ConceptNetLoaderConfig()
        super().__init__(config=config)

    def __post_init__(self) -> None:
        self._download_url = self.config.download_urls[self.config.version]
        fn = self._download_url.split("/")[-1]
        self.csv_gz_path = (
            Path(kblite_config.resources_dir) / self.config.identifier / "loader" / fn
        )
        self.jsonl_gz_path = self.csv_gz_path.with_suffix(".jsonl.gz")
        super().__post_init__()

    def download(self) -> None:
        """Download resource."""
        if self.csv_gz_path.exists():
            logger.info("File already exists: %s", self.csv_gz_path)
            return
        self.csv_gz_path.parent.mkdir(parents=True, exist_ok=True)
        url = self._download_url
        logger.info("Downloading %s to %s", url, self.csv_gz_path)
        with self.csv_gz_path.open("wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get("content-length")
            if total_length is None:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
        logger.info("Download complete.")

    def _read_from_csv(self, n_rows: int) -> Generator[str, None, None]:
        pbar = tqdm.tqdm(
            total=n_rows,
            desc="Reading the CSV file and writing to JSONL",
            disable=self.config.verbose < 1,
        )
        with gzip.open(self.csv_gz_path, "rt") as fr:
            for line in fr:
                yield csv.read_line(line)
                pbar.update(1)
        pbar.close()

    def write_to_jsonl(self) -> int:
        """Write the CSV file to a JSONL file."""
        with gzip.open(self.csv_gz_path, "rt") as f:
            n_rows_csv = sum(1 for _ in f)
        # clean up the jsonl file if it exists or the row count differs
        jsonl_gz_path = self.jsonl_gz_path
        if jsonl_gz_path.exists():
            with gzip.open(jsonl_gz_path, "rt") as f:
                n_rows_jsonl = sum(1 for _ in f)
            if n_rows_csv == n_rows_jsonl:
                logger.info("File already exists: %s", jsonl_gz_path)
                return n_rows_csv
            else:
                logger.warning(
                    "Overwriting '%s' since the row count differs (CSV: %d, JSONL: %d)",
                    jsonl_gz_path,
                    n_rows_csv,
                    n_rows_jsonl,
                )
                jsonl_gz_path.unlink()
                self.jsonl_gz_path.parent.mkdir(parents=True, exist_ok=True)
                write_to_file(
                    self.jsonl_gz_path, self._read_from_csv(n_rows=n_rows_csv)
                )
                logger.info("Writing to %s complete.", self.jsonl_gz_path)
        return n_rows_csv

    def iterrows(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """Iterate over edges."""
        self.download()
        n_rows = self.write_to_jsonl()
        pbar = tqdm.tqdm(
            total=n_rows,
            desc="Reading the JSONL file",
            disable=self.config.verbose < 1,
        )
        with gzip.open(self.jsonl_gz_path, "rt") as f:
            for i, line in enumerate(f):
                yield i, orjson.loads(line)
                if pbar:
                    pbar.update(1)
        pbar.close()

    def count(self) -> int:
        with gzip.open(self.jsonl_gz_path, "rt") as f:
            return sum(1 for _ in f)
