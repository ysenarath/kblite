from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import ForeignKey
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)
from typing_extensions import Self


def apply_prefix(uri: str | None, namespace: str | None = None) -> str:
    if uri is None:
        return None
    if namespace and uri.startswith("/"):
        namespace = namespace.rstrip("/")
        return f"{namespace}{uri}"
    return uri


class Base(DeclarativeBase):
    pass


class Node(Base):
    __tablename__ = "node"

    id: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[str | None] = mapped_column(nullable=True)
    language: Mapped[str | None]
    sense_label: Mapped[str | None]
    term_id: Mapped[str | None] = mapped_column(ForeignKey("node.id"))
    site: Mapped[str | None]
    path: Mapped[str | None]
    site_available: Mapped[bool | None]

    term: Mapped[Node | None] = relationship()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Session | None = None,
        commit: bool = True,
        namespace: str | None = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        if session:
            instance = session.get(cls, id_)
            if instance:
                return instance
        term_id = data.get("term")
        if term_id:
            term_id = apply_prefix(term_id, namespace=namespace)
        instance = cls(
            id=id_,
            label=data["label"],
            language=data.get("language"),
            sense_label=data.get("sense_label"),
            term_id=term_id,
            site=data.get("site"),
            path=data.get("path"),
            site_available=data.get("site_available"),
        )
        if not session:
            return instance
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def __repr__(self) -> str:
        return f"Node(id='{self.id}', label='{self.label}', language='{self.language}')"


class Relation(Base):
    __tablename__ = "relation"

    id: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[str]
    symmetric: Mapped[bool] = mapped_column(default=False)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Session | None = None,
        commit: bool = True,
        namespace: str | None = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        if session:
            instance = session.get(cls, id_)
            if instance:
                return instance
        instance = cls(
            id=id_,
            label=data["label"],
            symmetric=data.get("symmetric", False),
        )
        if not session:
            return instance
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def __repr__(self) -> str:
        return f"Relation(id='{self.id}', label='{self.label}', symmetric={self.symmetric})"


class Source(Base):
    __tablename__ = "source"

    id: Mapped[str] = mapped_column(primary_key=True)
    contributor: Mapped[str | None]
    process: Mapped[str | None]
    activity: Mapped[str | None]
    edge_id: Mapped[str] = mapped_column(ForeignKey("edge.id"))

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Session | None = None,
        commit: bool = True,
        namespace: str | None = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        edge_id = data["edge_id"]  # assume this is already prefixed
        if session:
            instance = session.get(cls, id_)
            if instance:
                return instance
        kwargs = {}
        for key in ("contributor", "process", "activity"):
            if key not in data:
                continue
            value = data[key]
            kwargs[key] = apply_prefix(value, namespace=namespace)
        instance = cls(id=id_, edge_id=edge_id, **kwargs)
        if not session:
            return instance
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def __repr__(self) -> str:
        return f"Source(id='{self.id}', contributor='{self.contributor}')"


class Edge(Base):
    __tablename__ = "edge"

    id: Mapped[str] = mapped_column(primary_key=True)
    rel_id: Mapped[str] = mapped_column(ForeignKey("relation.id"), index=True)
    start_id: Mapped[str] = mapped_column(ForeignKey("node.id"), index=True)
    end_id: Mapped[str] = mapped_column(ForeignKey("node.id"), index=True)
    license: Mapped[str | None]
    weight: Mapped[float] = mapped_column(default=1.0)
    dataset: Mapped[str | None]
    surface_text: Mapped[str | None]

    rel: Mapped[Relation] = relationship()
    start: Mapped[Node] = relationship(foreign_keys=[start_id])
    end: Mapped[Node] = relationship(foreign_keys=[end_id])
    sources: Mapped[list[Source]] = relationship()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Session | None = None,
        commit: bool = True,
        namespace: str | None = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        if session:
            instance = session.get(cls, id_)
            if instance:
                return instance
        rel = Relation.from_dict(
            data["rel"], session=session, commit=commit, namespace=namespace
        )
        start = Node.from_dict(
            data["start"], session=session, commit=commit, namespace=namespace
        )
        end = Node.from_dict(
            data["end"], session=session, commit=commit, namespace=namespace
        )
        sources = [
            Source.from_dict(
                {"edge_id": id_, **source},
                session=session,
                commit=commit,
                namespace=namespace,
            )
            for source in data["sources"]
        ]
        dataset = apply_prefix(data.get("dataset"), namespace=namespace)
        instance = cls(
            id=id_,
            rel=rel,
            start=start,
            end=end,
            sources=sources,
            license=data["license"],
            weight=data["weight"],
            dataset=dataset,
            surface_text=data.get("surfaceText", data.get("surface_text")),
        )
        if not session:
            return instance
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def __repr__(self) -> str:
        return f"Edge(id='{self.id}', start='{self.start_id}', rel='{self.rel_id}', end='{self.end_id}')"


@dataclass
class Feature:
    rel: Relation | None = None
    start: Node | None = None
    end: Node | None = None
    node: Node | None = None


@dataclass
class RelatedNode:
    id: str
    weight: float


@dataclass
class PartialCollectionView:
    paginatedProperty: str
    firstPage: str
    nextPage: str | None = None
    previousPage: str | None = None


@dataclass
class Query:
    id: str
    edges: list[Edge] | None = None
    features: list[Feature] | None = None
    related: list[RelatedNode] | None = None
    view: PartialCollectionView | None = None
    value: float | None = None
    license: str | None = None
