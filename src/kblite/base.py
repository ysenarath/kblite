from __future__ import annotations

import os
import shutil
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Set

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import auto as tqdm
from huggingface_hub import snapshot_download

from .vocab import Vocab
from .loader import KnowledgeLoader
from .models import Edge, Node
from .triplet import TripletStore

# from .triplet import NodeIndex, TripletStore

__all__ = [
    "KnowledgeBase",
]

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        database_name_or_path: str | Path,
        file_name: str = "conceptnet-v5.7.0",
        verbose: int = 1,
    ):
        # output path with checksum
        self.path = Path(database_name_or_path) / "data" / f"{file_name}.db"
        self.engine = create_engine(f"sqlite:///{self.path}", echo=verbose > 2)
        with self.session() as session:
            session.execute(text("PRAGMA journal_mode=WAL"))
        # create index if not exists
        self.index = self._get_or_create_index()
        # self.node_index = NodeIndex(self.path.with_suffix(".node.db"))

    def _get_or_create_index(self) -> TripletStore:
        # populate index if not frozen (frozen means index is up-to-date)
        index_path = str(self.path.with_suffix("")) + "-index"
        index = TripletStore(index_path)
        if index.frozen:
            return index
        if os.path.exists(index_path):
            print(f"Removing existing index at {index_path}")
            shutil.rmtree(index_path, ignore_errors=True)
        index.close()
        del index
        index = TripletStore(index_path)
        with self.session() as session:
            n_total = session.query(Edge).count()
            query = session.query(Edge.start_id, Edge.rel_id, Edge.end_id)
            pbar = tqdm.tqdm(query.yield_per(int(1e5)), total=n_total, desc="Indexing")
            index.add(pbar)
        index.frozen = True
        return index

    def get_node_ids_by_label(self, label: str) -> Iterable[str]:
        return self.label2index.get(label, set())

    def _create_label2index(self) -> dict[str, Set[int]]:
        label2index: dict[str, Set[int]] = {}
        with self.session() as session:
            for node in session.query(Node):
                if node.label not in label2index:
                    label2index[node.label] = set()
                label2index[node.label].add(node.id)
        return label2index

    def num_edges(self) -> int:
        with self.session() as session:
            return session.query(Edge).count()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        yield Session(self.engine)

    def cleanup(self):
        if not os.path.exists(self.path):
            return
        os.remove(self.path)

    def iternodes(self, verbose: bool = False) -> Iterable[Node]:
        with self.session() as session:
            query = session.query(Node).yield_per(1000)
            total = session.query(Node).count()
            pbar = tqdm.tqdm(
                query, desc="Iterating Nodes", disable=not verbose, total=total
            )
            yield from pbar

    def get_vocab(self) -> Vocab:
        with self.session() as session:
            n_total = session.query(Node).count()
        # populate vocab if not frozen (frozen means vocab is up-to-date)
        vocab_db_path = self.path.with_name(self.path.stem + "-vocab.db")
        config = {
            "type": "sqlalchemy",
            "url": f"sqlite:///{vocab_db_path}",
        }
        if os.path.exists(vocab_db_path):
            vocab = Vocab(config)
            return vocab
        vocab_db_temp_path = self.path.with_name(self.path.stem + "-vocab.db.tmp")
        # remove the existing vocab database
        if os.path.exists(vocab_db_temp_path):
            os.remove(vocab_db_temp_path)
        tmp_config = {
            "type": "sqlalchemy",
            "url": f"sqlite:///{vocab_db_temp_path}",
        }
        vocab = Vocab(tmp_config)
        batch_size = int(1e4)
        with self.session() as session:
            query = session.query(Node)
            pbar = tqdm.tqdm(
                query.yield_per(batch_size),
                total=n_total,
                desc="Building Vocabulary",
            )
            nodes = []
            for i, node in enumerate(pbar):  # this will be done in parallel
                nodes.append(node)
                if i % batch_size == 0:
                    vocab.extend(nodes)
                    nodes = []
            if nodes:
                vocab.extend(nodes)
        del vocab
        # move the temporary database to the final location
        shutil.move(vocab_db_temp_path, vocab_db_path)
        vocab = Vocab(config)
        return vocab

    @classmethod
    def from_loader(cls, loader: KnowledgeLoader) -> KnowledgeBase:
        identifier = loader.config.identifier
        version = loader.config.version or "0.0.1"
        if not identifier.isidentifier():
            raise ValueError("identifier must be a valid Python identifier")
        database_name = os.path.join(identifier, f"{identifier}-v{version}")
        self = cls(database_name, create=True)
        with self.session() as session:
            session.execute(text("PRAGMA journal_mode=WAL"))
            for i, val in enumerate(loader.iterrows()):
                _ = Edge.from_dict(  # ignore the return value
                    val,
                    session=session,
                    commit=False,
                    namespace=loader.config.namespace,
                )
                if i % 100 == 0:
                    try:
                        session.commit()
                    except Exception as e:
                        session.rollback()
                        raise e
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    @staticmethod
    def from_repository(repo_id: str) -> KnowledgeBase:
        """Download and load a pre-trained index"""
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
        )
        return KnowledgeBase(snapshot_path)
