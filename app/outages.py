"""Admin-declared outages of provider instances and single models.

An outage takes its models out of service for every caller, admins and allow-mode
profiles included: app.model_resolution evaluates them as "outage", which picks the
fallback mappings, and answers 503 (with the admin's description) when no fallback
serves the request. The models stay listed, so clients see the same catalogue.
Calls on an existing response (/v1/responses/{id} retrieve, cancel, delete,
input_items) follow its stored provider and are not refused: they start no new
generation.

Reads never touch the database: the registry holds an immutable snapshot, reloaded
after every admin change and every 30 seconds, so other workers catch up too.

Layering: imported by app.model_resolution, app.rate_limit_dep and the routes, so
it must not import any of them (or app.providers) at module scope.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL_SECONDS = 30


@dataclass(frozen=True)
class Outage:
    id: int
    scope: str          # "provider" | "model"
    target: str         # provider_key or model_id
    description: str
    created_at: Optional[datetime] = None


class OutageRegistry:
    """Immutable-snapshot outage holder; reads never touch the database."""

    def __init__(self) -> None:
        # (model_id -> Outage, provider_key -> Outage), swapped whole.
        self._state: Tuple[Dict[str, Outage], Dict[str, Outage]] = ({}, {})
        self._refresh_task: Optional[asyncio.Task] = None

    def install(self, outages: Iterable[Outage]) -> None:
        """Replace the whole snapshot (used by load_from_database, and by tests)."""
        by_model: Dict[str, Outage] = {}
        by_provider: Dict[str, Outage] = {}
        for outage in outages:
            (by_provider if outage.scope == "provider" else by_model)[outage.target] = outage
        self._state = (by_model, by_provider)

    def reset(self) -> None:
        self._state = ({}, {})

    def outage_for(self, model_id: Optional[str]) -> Optional[Outage]:
        """The outage covering `model_id`: its own, else its provider instance's."""
        if not model_id:
            return None
        by_model, by_provider = self._state
        outage = by_model.get(model_id)
        if outage is not None or '/' not in model_id:
            return outage
        return by_provider.get(model_id.split('/', 1)[0])

    def is_in_outage(self, model_id: Optional[str]) -> bool:
        return self.outage_for(model_id) is not None

    def provider_outage(self, provider_key: str) -> Optional[Outage]:
        return self._state[1].get(provider_key)

    def all(self) -> List[Outage]:
        by_model, by_provider = self._state
        return [*by_provider.values(), *by_model.values()]

    async def load_from_database(self) -> None:
        from app.auth.database import AsyncSessionLocal, get_all_outages

        async with AsyncSessionLocal() as db:
            rows = await get_all_outages(db)
        self.install(
            Outage(id=row.id, scope=row.scope, target=row.target,
                   description=row.description, created_at=row.created_at)
            for row in rows
        )

    async def start(self) -> None:
        await self.load_from_database()
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        task, self._refresh_task = self._refresh_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)
            try:
                await self.load_from_database()
            except Exception:
                logger.warning("Outage refresh failed", exc_info=True)


def describe(outages: Iterable[Outage]) -> str:
    """The outages' descriptions, deduplicated in order, for an error message."""
    seen: List[str] = []
    for outage in outages:
        if outage.description not in seen:
            seen.append(outage.description)
    return "; ".join(seen)


outage_registry = OutageRegistry()
