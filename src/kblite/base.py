from __future__ import annotations

import contextvars
from dataclasses import dataclass

from sqlalchemy.orm import Session

var = contextvars.ContextVar["SessionContext"]("session_var")
var.set(None)


@dataclass
class SessionContext:
    session: Session
    partial_commit: bool
    namespace: str


def get_context_vars() -> tuple[Session | None, bool, str]:
    session_ctx = var.get()
    if session_ctx is None:
        return None, False, "http://example.com/"
    return session_ctx.session, session_ctx.partial_commit, session_ctx.namespace


def apply_prefix(uri: str | None) -> str:
    _, _, namespace = get_context_vars()
    if uri is None:
        return None
    if namespace and uri.startswith("/"):
        namespace = namespace.rstrip("/")
        return f"{namespace}{uri}"
    return uri
