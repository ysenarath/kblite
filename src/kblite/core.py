from __future__ import annotations

import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List

import orjson
from nightjar import BaseConfig, BaseModule
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from kblite.base import SessionContext, var
from kblite.config import config as kblite_config
from kblite.loader import AutoKnowledgeBaseLoader, KnowledgeBaseLoaderConfig
from kblite.models import Base, Edge, Node

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
        self.vocab_path = resources_dir / identifier / f"vocab-v{version}.json"

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
        with Session(self.bind) as session:
            ctx = SessionContext(session, partial_commit, self.config.loader.namespace)
            token = var.set(ctx)
            yield session
            var.reset(token)

    def get_vocab(self, force: bool = False) -> Dict[str, List[str]]:
        if self.vocab_path.exists():
            if force:
                self.vocab_path.unlink()
            else:
                with open(self.vocab_path, "r") as fp:
                    return orjson.loads(fp.read())
        vocab = defaultdict(set)
        with self.session() as session:
            results = session.query(Node.label, Node.id).distinct().all()
            for label, id in results:
                vocab[label].add(id)
        # convert sets to lists
        vocab = {label: list(ids) for label, ids in vocab.items()}
        # write to file (json)
        with open(self.vocab_path, "w") as fp:
            json.dump(vocab, fp)
        return vocab
