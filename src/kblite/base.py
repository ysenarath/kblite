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
