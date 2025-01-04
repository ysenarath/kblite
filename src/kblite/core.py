from __future__ import annotations

import functools
import json
import logging
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, cast

import marisa_trie as mt
import numpy as np
import tqdm
from fasttext.FastText import _FastText as FastText
from nightjar import BaseConfig, BaseModule
from rapidfuzz import fuzz, process
from sqlalchemy import alias, create_engine
from sqlalchemy.orm import Session

from kblite.base import SessionContext, var
from kblite.config import config as kblite_config
from kblite.loader import (
    AutoEmbeddingLoader,
    AutoKnowledgeBaseLoader,
    EmbeddingLoaderConfig,
    KnowledgeBaseLoaderConfig,
)
from kblite.models import Base, Edge, Node, Relation

logger = logging.getLogger(__name__)


class KnowledgeBaseConfig(BaseConfig):
    loader: KnowledgeBaseLoaderConfig


class KnowledgeBase(BaseModule):
    config: KnowledgeBaseConfig

    def __post_init__(self):
        resources_dir = Path(kblite_config.resources_dir)
        identifier = self.config.loader.identifier
        version = self.config.loader.version
        path = resources_dir / identifier / f"database-v{version}.sqlite"
        url = f"sqlite:///{path}"
        self.bind = create_engine(url)
        self.create_all()

    def create_all(self):
        Base.metadata.create_all(self.bind)

    def populate(self):
        loader = AutoKnowledgeBaseLoader(self.config.loader)
        with self.session() as session:
            # get number of rows in the table
            existing_count = session.query(Edge).count()
            # get number of edges
            total_count = loader.count()
            # db fingerprint
            if total_count == existing_count:
                logger.info("Database is up-to-date. Skipping loading.")
                return
            to_commit = False
            for i, row in loader.iterrows():
                Edge.from_dict(row)
                to_commit = True
                if i % 1000 == 0 and to_commit:
                    session.commit()
                    to_commit = False
            if to_commit:
                session.commit()

    @contextmanager
    def session(self, partial_commit: bool = False) -> Generator[Session, None, None]:
        current = var.get()
        if current:
            raise RuntimeError("session already in progress")
        with Session(self.bind) as session:
            ctx = SessionContext(session, partial_commit, self.config.loader.namespace)
            token = var.set(ctx)
            yield session
            var.reset(token)

    def rebuild(self):
        self.vocab.rebuild()
        self.triplets.rebuild()

    @property
    def vocab(self) -> Vocab:
        if getattr(self, "_vocab", None) is None:
            self._vocab = Vocab(self)
        return self._vocab

    @property
    def triplets(self) -> Triplets:
        if getattr(self, "_triplets", None) is None:
            self._triplets = Triplets(self)
        return self._triplets


class Vocab:
    def __init__(self, kb: KnowledgeBase):
        self._get_kb: KnowledgeBase = weakref.ref(kb)
        self._trie: Optional[mt.BytesTrie] = None
        self.lang: str = "en"

    @property
    def kb(self) -> KnowledgeBase:
        return self._get_kb()

    @property
    def trie(self) -> mt.BytesTrie:
        if self._trie is None:
            self._load_trie()
        return self._trie

    def _get_trie_path(self) -> Path:
        resources_dir = Path(kblite_config.resources_dir)
        identifier = self.kb.config.loader.identifier
        version = self.kb.config.loader.version
        return resources_dir / identifier / f"vocab-{self.lang}-v{version}.marisa"

    def _load_trie(self, force: bool = False) -> None:
        trie_path = self._get_trie_path()
        if not force and trie_path.exists():
            self._trie = mt.BytesTrie().mmap(str(trie_path))
        else:
            self._build_trie()

    def _iter_from_kb(self) -> Generator[Tuple[str, bytes], None, None]:
        with self.kb.session() as session:
            nodes = session.query(Node.label, Node.id).distinct()
            if self.lang:
                nodes = nodes.filter(
                    Node.language == self.lang,
                )
            pbar = tqdm.tqdm(
                nodes.yield_per(1000), total=nodes.count(), desc="Building trie"
            )
            for label, node_id in pbar:
                label = label.strip().lower()
                if not label:
                    continue
                yield label, node_id.encode()

    def _build_trie(self) -> None:
        """Build trie from nodes in the database"""
        logger.info("Building vocabulary trie...")
        trie_path = self._get_trie_path()
        if trie_path.exists():
            trie_path.unlink()
        self._trie = mt.BytesTrie(self._iter_from_kb())
        self._trie.save(str(trie_path))
        logger.info("Vocabulary trie built and saved")

    @functools.lru_cache(maxsize=10000)
    def _getitem_cached(self, key: str) -> List[str]:
        """Get all node IDs for a given label"""
        values = self.trie[key.strip().lower()]
        return list(map(bytes.decode, values))

    def keys(self) -> List[str]:
        return self.trie.keys()

    def items(self) -> Generator[Tuple[str, List[str]], None, None]:
        for key in self.trie.keys():
            yield key, self._getitem_cached(key)

    def __getitem__(self, key: str) -> List[str]:
        try:
            return self._getitem_cached(key)
        except Exception:
            raise KeyError(key)

    def __contains__(self, label: str) -> bool:
        return label in self.trie

    def __len__(self) -> int:
        return len(self.trie)

    def rebuild(self) -> None:
        """Force rebuild the trie"""
        if self._trie:
            self._trie = None
        self._load_trie(force=True)
        self._getitem_cached.cache_clear()  # clear the cache

    @functools.lru_cache(maxsize=10000)
    def startswith(self, prefix: str) -> List[str]:
        """Get all node IDs for a given prefix"""
        prefix = prefix.strip().lower()
        return self.trie.keys(prefix)

    def find(self, text: str, fuzzy: bool | dict = False) -> List[str]:
        """Find node IDs for a given label"""
        if fuzzy is not False:
            if isinstance(fuzzy, dict):
                return self.find_fuzzy(text, **fuzzy)
            return self.find_fuzzy(text)
        return self._getitem_cached(text)

    def find_fuzzy(self, text: str, limit: int = 5) -> List[str]:
        """Find node IDs for a given label using fuzzy matching"""
        text = text.strip().lower()
        if not hasattr(self, "_choices"):
            self._choices = set(self.trie.keys())
        results = process.extract(
            text,
            self._choices,
            scorer=fuzz.QRatio,
            limit=limit,
        )
        return results


class Triplets:
    def __init__(self, kb: KnowledgeBase):
        self._get_kb: KnowledgeBase = weakref.ref(kb)
        self._trie: Optional[mt.BytesTrie] = None
        self.lang = "en"

    @property
    def kb(self) -> KnowledgeBase:
        return self._get_kb()

    @property
    def trie(self) -> mt.BytesTrie:
        if self._trie is None:
            self._load_trie()
        return self._trie

    def _get_trie_path(self) -> Path:
        resources_dir = Path(kblite_config.resources_dir)
        identifier = self.kb.config.loader.identifier
        version = self.kb.config.loader.version
        return resources_dir / identifier / f"triplets-{self.lang}-v{version}.marisa"

    def _load_trie(self, force: bool = False) -> None:
        trie_path = self._get_trie_path()
        if not force and trie_path.exists():
            self._trie = mt.BytesTrie().mmap(str(trie_path))
        else:
            self._build_trie()

    def _build_trie(self) -> None:
        """Build trie from nodes in the database"""
        logger.info("Building triplets trie...")
        trie_path = self._get_trie_path()
        if trie_path.exists():
            trie_path.unlink()
        self._trie = mt.BytesTrie(self._iter_from_kb())
        self._trie.save(str(trie_path))
        logger.info("Triplets trie built and saved")

    def _iter_from_kb(self) -> Generator[Tuple[str, bytes], None, None]:
        with self.kb.session() as session:
            start_node = alias(Node, "start_node")
            end_node = alias(Node, "end_node")
            edges = (
                session.query(
                    *[
                        c
                        for c in (
                            start_node.c.label,
                            Relation.label,
                            end_node.c.label,
                            Edge.id,
                        )
                    ]
                )
                .select_from(Edge)
                .join(start_node, Edge.start_id == start_node.c.id)
                .join(Relation, Edge.rel_id == Relation.id)
                .join(end_node, Edge.end_id == end_node.c.id)
            )
            if self.lang:
                edges = edges.filter(
                    cast(Node, start_node.c).language == self.lang,
                    cast(Node, end_node.c).language == self.lang,
                )
            edges = edges.distinct()
            pbar = tqdm.tqdm(
                edges.yield_per(1000), total=edges.count(), desc="Building trie"
            )
            for start, rel, end, id in pbar:
                start = start.strip().lower().replace("\t", " ")
                end = end.strip().lower().replace("\t", " ")
                if not start or not rel or not end:
                    continue
                yield f"spo\t{start}\t{rel}\t{end}", str(id).encode()
                yield f"ops\t{end}\t{rel}\t{start}", str(id).encode()

    def rebuild(self) -> None:
        """Force rebuild the trie"""
        if self._trie:
            self._trie = None
        self._load_trie(force=True)
        self.find.cache_clear()  # clear the cache

    @functools.lru_cache(maxsize=10000)
    def find(self, subj: Optional[str] = None, rel: Optional[str] = None) -> List[str]:
        """Find edge IDs for a given triplet"""
        if subj:
            subj = subj.strip().lower().replace("\t", " ")
            if rel:
                return self.parse(self.trie.keys(f"spo\t{subj}\t{rel}\t"))
            return self.parse(self.trie.keys(f"spo\t{subj}\t"))
        return []

    def camel_to_natural(self, text: str) -> str:
        if not text:
            return text
        result = text[0]
        for char in text[1:]:
            if char.isupper():
                result += " " + char
            else:
                result += char
        return result.lower()

    def parse(self, keys: List[str]) -> List[Tuple[str, str, str]]:
        return_ = []
        for key in keys:
            _, start, rel, end = key.split("\t")
            rel = self.camel_to_natural(rel)
            return_.append((start, rel, end))
        return return_

    def __iter__(self) -> Generator[Tuple[str, str, str], None, None]:
        for key in self.trie.keys(f"spo\t"):
            _, start, rel, end = key.split("\t")
            yield start, rel, end

    def __len__(self) -> int:
        return len(self.trie.keys("spo\t"))


class Embedding(Mapping):
    entity2id: Dict[str, int]
    vectors: np.ndarray
    model: Optional[FastText]

    def __init__(self, config: EmbeddingLoaderConfig | Mapping[str, Any]):
        if not isinstance(config, EmbeddingLoaderConfig):
            config = EmbeddingLoaderConfig.from_dict(config)
        self.config = config
        cache_path = (
            Path(kblite_config.resources_dir)
            / config.identifier
            / "data"
            / f"vectors-{config.version}.data"
        )
        # create embedding if not exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        loader = AutoEmbeddingLoader(config)
        try:
            self._load(cache_path)
        except FileNotFoundError:
            mapping = loader.load()
            self._from_dict(mapping)
            self._dump(cache_path)
        try:
            self.model = loader.get_model()
        except NotImplementedError:
            self.model = None

    def _from_dict(self, entitity_vectors: Dict[str, np.ndarray]) -> Embedding:
        entity2id = {entity: i for i, entity in enumerate(entitity_vectors.keys())}
        vectors = np.array(list(entitity_vectors.values()))
        self.entity2id = entity2id
        self.vectors = vectors

    def _dump(self, path: Path | str) -> None:
        path = Path(path)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        with open(path.with_suffix(".entity2id.json"), "w") as f:
            json.dump(self.entity2id, f)

    def _load(self, path: Path | str) -> Embedding:
        path = Path(path)
        with open(path.with_suffix(".entity2id.json")) as f:
            entity2id = json.load(f)
        vectors: np.ndarray = np.load(path.with_suffix(".vectors.npy"))
        # memory-mapped array
        vectors = np.memmap(
            path.with_suffix(".vectors.npy"),
            dtype=np.float32,
            mode="r",
            shape=vectors.shape,
        )
        self.entity2id = entity2id
        self.vectors = vectors

    def get_word_vector(self, word: str) -> np.ndarray:
        if self.model:
            return self.model.get_word_vector(word)
        return self[word]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.entity2id[key]]

    def __iter__(self) -> Iterable[str]:
        return iter(self.entity2id)

    def __len__(self) -> int:
        return len(self.entity2id)
