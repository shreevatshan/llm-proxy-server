"""Pytest fixtures shared across the test suite.

Disposes the SQLAlchemy engines at the end of the test session so the
interpreter can exit cleanly.

Why this is needed: ``app.auth.database`` creates a module-level
``sqlite+aiosqlite`` async engine whose connection pool holds an open
aiosqlite connection. aiosqlite runs each connection on a worker thread that
is **not** a daemon thread (see aiosqlite ``core.py`` — ``Thread(...)`` with no
``daemon=True``), so it only terminates when the connection is closed. Nothing
in the tests disposes the engine, so that non-daemon worker thread stays alive
and blocks interpreter shutdown — pytest reports all tests passing, then the
process hangs in ``threading._shutdown``. Disposing the engine closes the
pooled connection, which stops the worker thread.
"""

import asyncio

import pytest


@pytest.fixture(scope="session", autouse=True)
def _dispose_db_engines():
    """Dispose the async + sync DB engines after the session ends."""
    yield

    try:
        from app.auth.database import engine, sync_engine
    except Exception:
        return

    # Dispose the async engine on a fresh event loop — this closes the pooled
    # aiosqlite connection and stops its non-daemon worker thread.
    try:
        asyncio.run(engine.dispose())
    except Exception:
        pass

    try:
        sync_engine.dispose()
    except Exception:
        pass
