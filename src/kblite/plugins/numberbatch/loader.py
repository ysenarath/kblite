import gzip
import logging
from pathlib import Path
from typing import ClassVar, Dict

import numpy as np
import requests
import tqdm
from typing_extensions import Self

from kblite.config import config as kblite_config
from kblite.loader import EmbeddingLoader, EmbeddingLoaderConfig

URLS = {
    "numberbatch-19.08": "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz"
}

logger = logging.getLogger(__name__)


class NumberbatchLoaderConfig(EmbeddingLoaderConfig):
    identifier: ClassVar[str] = "numberbatch"
    version: str = "19.08"
    # use http:// since we used it in conceptnet also
    namespace: str = "http://conceptnet.io/"


class NumberbatchLoader(EmbeddingLoader):
    config: NumberbatchLoaderConfig

    def __post_init__(self):
        super().__post_init__()
        self.url = URLS[f"numberbatch-{self.config.version}"]
        filename = self.url.split("/")[-1]
        self.embeddings_path = (
            Path(kblite_config.resources_dir)
            / self.config.identifier
            / "loader"
            / filename
        )

    def download(self) -> Self:
        url = self.url
        embeddings_path = self.embeddings_path
        if embeddings_path.exists():
            logger.info("File already exists: %s", embeddings_path)
            return self
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", url, embeddings_path)
        with embeddings_path.open("wb") as f:
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
        return self

    def parse_raw_file(self):
        with gzip.open(self.embeddings_path, "rt") as f:
            num_rows = sum(1 for _ in f)
        entitity_vectors = {}  # node id -> vector
        with gzip.open(self.embeddings_path, "rt") as f:
            for line in tqdm.tqdm(f, total=num_rows):
                parts = line.split(" ")
                vec = list(map(float, parts[1:]))
                if len(vec) != 300:
                    continue
                id_ = self.config.namespace.rstrip("/") + "/" + parts[0].lstrip("/")
                entitity_vectors[id_] = vec
        return entitity_vectors

    def load(self) -> Dict[str, np.ndarray]:
        return self.download().parse_raw_file()
