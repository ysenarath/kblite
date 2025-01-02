from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Dict

import fasttext
import fasttext.util
import numpy as np
from typing_extensions import Self

from kblite.config import config as kblite_config
from kblite.loader import EmbeddingLoader, EmbeddingLoaderConfig

URLS = {"fasttext-cc.en.300": None}

logger = logging.getLogger(__name__)


@contextmanager
def change_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


class FastTextLoaderConfig(EmbeddingLoaderConfig):
    identifier: ClassVar[str] = "fasttext"
    version: str = "cc.en.300"


class FastTextLoader(EmbeddingLoader):
    config: FastTextLoaderConfig

    def __post_init__(self):
        super().__post_init__()
        self.files_dir = (
            Path(kblite_config.resources_dir) / self.config.identifier / "loader"
        )

    def download(self) -> Self:
        lang_id = self.config.version.split(".")[1]
        with change_dir(self.files_dir):
            fasttext.util.download_model(lang_id, if_exists="ignore")
        return self

    def read(self) -> Dict[str, np.ndarray]:
        fn = f"{self.config.version}.bin"
        model = fasttext.load_model(f"{self.files_dir / fn}")
        vectors = {}
        for term in model.get_words():
            vectors[term] = model.get_word_vector(term)
        return vectors

    def load(self) -> Dict[str, np.ndarray]:
        return self.download().read()
