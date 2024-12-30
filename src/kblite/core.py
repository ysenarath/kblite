from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from nightjar import BaseConfig, BaseModule
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from kblite.base import SessionContext, var
from kblite.config import config as kblite_config
from kblite.loader import AutoKnowledgeBaseLoader, KnowledgeBaseLoaderConfig
from kblite.models import Base, Edge


class KnowledgeBaseConfig(BaseConfig):
    loader: KnowledgeBaseLoaderConfig


class KnowledgeBase(BaseModule):
    config: KnowledgeBaseConfig

    def __post_init__(self):
        path = (
            Path(kblite_config.resources_dir)
            / self.config.loader.identifier
            / "database.sqlite"
        )
        url = f"sqlite:///{path}"
        self.bind = create_engine(url)
        self.create_all()
        loader = AutoKnowledgeBaseLoader(self.config.loader)
        with self.session() as session:
            for i, row in enumerate(loader.iterrows()):
                Edge.from_dict(row)
                if i % 1000 == 0:
                    session.commit()
            session.commit()

    def create_all(self):
        Base.metadata.create_all(self.bind)

    @contextmanager
    def session(self, partial_commit: bool = False) -> Generator[Session, None, None]:
        with Session(self.bind) as session:
            ctx = SessionContext(session, partial_commit, self.config.loader.namespace)
            token = var.set(ctx)
            yield session
            var.reset(token)
