"""Request-pool settlement: charging members for what the pool spent while they were in it.

A pool shares one daily request quota (RPD). Its limit is the sum of its members'
individual limits and its count is the sum of their consumption. The subtle part is
*composition change* -- someone joining or leaving mid-day.

Letting each member walk away with their own raw row count would let a heavy consumer
carry a deficit into the next pool they join, and consume serially across pools far
beyond their own limit. So every change of composition **settles** the interval that
just closed: the requests the pool consumed since the previous change are divided among
the members who were present for them, in proportion to their limits, and banked into a
per-member running ``charged`` total.

    delta     = pool_used_now - SUM(charged_m)     # consumed since the last change
    share_m   = apportion(delta, by limit_m, capped at limit_m - charged_m)
    charged_m = charged_m + share_m
    carry_m   = charged_m - rows_m                 # absolute; effective_used == charged_m

Two properties follow directly, and they are the whole point:

* A member is **never charged for consumption that predates their arrival** -- that
  delta was closed out against the members who were actually there.
* A member is **never credited for consumption after their departure** -- their carry is
  frozen at the moment they leave.

``carry`` is absolute rather than incremental, so re-settling is idempotent and
self-correcting.

INVARIANT: after every composition change, ``SUM(charged_m) == pool_used_now`` over the
current members. It holds because ``carry_m == charged_m - rows_m``, so
``pool_used = SUM(rows_m + carry_m) = SUM(charged_m)``. That is why the pool's "usage at
the last settlement" is never stored -- it *is* ``SUM(charged)``.

Everything here is whole requests. ``request_usage.request_count`` is an integer and so
are limits, so no float ever enters the enforcement path; :func:`apportion` uses
largest-remainder apportionment to make integer splits sum exactly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MemberShare:
    """One member's position in a scope, as settlement sees it.

    ``limit`` must not be None: an unlimited member makes the whole pool unlimited on
    that scope, and settlement short-circuits before reaching apportionment (nothing is
    enforced, so nothing is owed).
    """
    user_id: int
    limit: int      # effective RPD for this scope; never None here
    charged: int    # what this member has already been charged today, >= 0
    consumed: int   # raw request_usage rows for this member/scope today; tie-break only


def apportion(delta: int, members: Sequence[MemberShare]) -> Dict[int, int]:
    """Split ``delta`` whole requests among ``members`` in proportion to their limits.

    Returns ``{user_id: share}``. Shares sum to exactly ``delta`` unless capping makes
    that impossible, in which case they sum to less -- quota is destroyed, never created.

    Largest-remainder (Hamilton) apportionment: floor each exact share, then hand out the
    leftover one request at a time by descending fractional part. Rounding each share
    independently would drift the total by up to N/2 requests per settlement, creating or
    destroying quota on every composition change.

    Ties go to the member who consumed more this interval, then to the lower user_id, so
    the heavy user absorbs the rounding and a light user is never rounded against.

    A member's share may not push their ``charged`` past their own ``limit``. Members who
    would exceed it are pinned at their remaining headroom and the split re-runs over the
    rest, at most len(members) times.

    A negative ``delta`` -- only reachable when usage rows are deleted underneath a live
    pool -- refunds proportionally by the same routine, with ``charged`` floored at 0.
    """
    if not members:
        return {}
    if delta == 0:
        return {m.user_id: 0 for m in members}

    refunding = delta < 0
    total = -delta if refunding else delta

    # Headroom is what each member can still absorb: unused limit when charging, and
    # what they have already been charged when refunding (you cannot refund past zero).
    headroom = {
        m.user_id: max(0, (m.limit - m.charged) if not refunding else m.charged)
        for m in members
    }
    weight = {m.user_id: max(0, m.limit) for m in members}
    consumed = {m.user_id: m.consumed for m in members}

    shares: Dict[int, int] = {m.user_id: 0 for m in members}
    active: List[int] = [m.user_id for m in members]
    remaining = total

    # Each pass either finishes or pins at least one member, so this runs at most
    # len(members) + 1 times.
    while active and remaining > 0:
        weight_sum = sum(weight[u] for u in active)
        if weight_sum == 0:
            # Nobody has a positive limit to apportion against; drop the remainder
            # rather than inventing a rule that would hand out quota arbitrarily.
            break

        floors = {u: (remaining * weight[u]) // weight_sum for u in active}
        leftover = remaining - sum(floors.values())
        if leftover:
            # Descending fractional part, then heavier consumer, then lower user_id.
            order = sorted(
                active,
                key=lambda u: (-((remaining * weight[u]) % weight_sum), -consumed[u], u),
            )
            for u in order[:leftover]:
                floors[u] += 1

        over = [u for u in active if floors[u] > headroom[u]]
        if not over:
            for u in active:
                shares[u] = floors[u]
            remaining = 0
            break

        # Pin the capped members at their headroom and re-apportion what is left.
        for u in over:
            shares[u] = headroom[u]
            remaining -= headroom[u]
            active.remove(u)

    if refunding:
        return {u: -s for u, s in shares.items()}
    return shares


# --------------------------------------------------------------------------- #
# Database settlement layer
# --------------------------------------------------------------------------- #


async def _load_members(db, pool_id: int) -> List[Tuple[int, str]]:
    """[(user_id, username)] for a pool, ordered by join time then id."""
    from sqlalchemy import select
    from app.auth.models import RequestPoolMember, User

    rows = (await db.execute(
        select(RequestPoolMember.user_id, User.username)
        .join(User, User.id == RequestPoolMember.user_id)
        .where(RequestPoolMember.pool_id == pool_id)
        .order_by(RequestPoolMember.joined_at, RequestPoolMember.id)
    )).all()
    return [(r.user_id, r.username) for r in rows]


def _fold_models_into_scopes(
    usage_rows: List[dict], id_of: Dict[str, int]
) -> Dict[Tuple[int, str, int], int]:
    """Fold per-model usage rows into (user_id, scope_kind, scope_id) -> request count.

    Each row lands in **exactly one** scope, because that is how enforcement counts it:
    ``get_today_count`` excludes any model that belongs to a model group or whose
    instance belongs to an instance group, since those are governed by the group's own
    limit. So a grouped row counts toward its group scope only -- instance group taking
    precedence, the same order enforcement resolves in -- and an ungrouped row counts
    toward 'overall'. Folding a grouped row into both would charge a member's overall
    quota for requests the overall gate never sees.

    Known imprecision: moving a model into a different group mid-day changes which scope
    its rows fold into, so carries written before the move describe a partition that no
    longer exists. It self-corrects at local midnight.
    """
    from app.rate_limit import rate_limit_tracker

    folded: Dict[Tuple[int, str, int], int] = {}
    for row in usage_rows:
        uid = id_of.get(row["user_identity"])
        if uid is None:
            continue
        count = int(row["request_count"])
        scope = rate_limit_tracker.scope_for_model(row["model"]) or ("overall", 0)
        key = (uid, scope[0], scope[1])
        folded[key] = folded.get(key, 0) + count
    return folded


async def _load_ledger(db, pool_id: int, today) -> Dict[Tuple[int, str, int], int]:
    """(user_id, scope_kind, scope_id) -> charged, for today. A missing row reads as 0."""
    from sqlalchemy import select
    from app.auth.models import RequestPoolLedger

    rows = (await db.execute(
        select(RequestPoolLedger).where(
            RequestPoolLedger.pool_id == pool_id,
            RequestPoolLedger.usage_date == today,
        )
    )).scalars().all()
    return {(r.user_id, r.scope_kind, r.scope_id): int(r.charged) for r in rows}


async def _load_carries(db, user_ids: Sequence[int], today) -> Dict[Tuple[int, str, int], int]:
    """(user_id, scope_kind, scope_id) -> carry, for today. A missing row reads as 0."""
    from sqlalchemy import select
    from app.auth.models import UserRpdCarry

    if not user_ids:
        return {}
    rows = (await db.execute(
        select(UserRpdCarry).where(
            UserRpdCarry.user_id.in_(list(user_ids)),
            UserRpdCarry.usage_date == today,
        )
    )).scalars().all()
    return {(r.user_id, r.scope_kind, r.scope_id): int(r.carry) for r in rows}


async def _upsert_ledger(db, pool_id: int, user_id: int, today, scope, charged: int) -> None:
    from sqlalchemy import select
    from app.auth.models import RequestPoolLedger

    kind, sid = scope
    row = (await db.execute(
        select(RequestPoolLedger).where(
            RequestPoolLedger.pool_id == pool_id,
            RequestPoolLedger.user_id == user_id,
            RequestPoolLedger.usage_date == today,
            RequestPoolLedger.scope_kind == kind,
            RequestPoolLedger.scope_id == sid,
        )
    )).scalar_one_or_none()
    if row is None:
        db.add(RequestPoolLedger(
            pool_id=pool_id, user_id=user_id, usage_date=today,
            scope_kind=kind, scope_id=sid, charged=int(charged),
        ))
    else:
        row.charged = int(charged)
    await db.flush()


async def _upsert_carry(db, user_id: int, today, scope, carry: int) -> None:
    from sqlalchemy import select
    from app.auth.models import UserRpdCarry

    kind, sid = scope
    row = (await db.execute(
        select(UserRpdCarry).where(
            UserRpdCarry.user_id == user_id,
            UserRpdCarry.usage_date == today,
            UserRpdCarry.scope_kind == kind,
            UserRpdCarry.scope_id == sid,
        )
    )).scalar_one_or_none()
    if row is None:
        if carry == 0:
            return  # a missing row already reads as 0; do not write noise
        db.add(UserRpdCarry(
            user_id=user_id, usage_date=today,
            scope_kind=kind, scope_id=sid, carry=int(carry),
        ))
    else:
        row.carry = int(carry)
    await db.flush()


async def _member_row_counts(db, members: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str, int], int]:
    """Today's raw request_usage counts per member per scope, read fresh.

    request_tracker buffers usage in memory and flushes every 60s, so a naive read can
    miss up to a minute of traffic -- and it would miss it in the worst direction, since
    the missing requests are exactly the ones a departing member most recently made.
    The caller holds pause_flush() around this so a concurrent flush cannot land between
    the read and the write and shift the counts underneath the settlement.
    """
    from app.auth.database import get_usage_by_user_and_model

    identities = [name for _, name in members]
    rows = await get_usage_by_user_and_model(db, identities, window="today")
    id_of = {name: uid for uid, name in members}
    return _fold_models_into_scopes(rows, id_of)


async def settle_pool(db, pool_id: int, *, dissolving: bool = False) -> None:
    """Close the current interval on every scope for one pool.

    Each member is charged their limit-share of what the pool consumed since the last
    composition change, and their carry is rewritten to ``charged - rows`` so that the
    rate limiter reads ``charged`` as their effective usage.

    Runs inside the caller's transaction and **before** the membership row is inserted or
    deleted, so the interval is closed against exactly the members who were present for
    it. Idempotent: settling again with nothing consumed in between rewrites the same
    numbers.
    """
    from contextlib import AsyncExitStack
    from app import time_utils
    from app.request_tracker import request_tracker
    from app.rate_limit import rate_limit_tracker

    members = await _load_members(db, pool_id)
    if not members:
        return

    today = time_utils.local_today()

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(request_tracker.pause_flush())
        await request_tracker.flush_pending()

        by_scope = await _member_row_counts(db, members)
        ledger = await _load_ledger(db, pool_id, today)
        carries = await _load_carries(db, [uid for uid, _ in members], today)

        for scope in rate_limit_tracker.settlement_scopes():
            kind, sid = scope
            limits = {
                uid: rate_limit_tracker.member_limit_for_scope(uid, kind, sid)
                for uid, _ in members
            }

            if any(v is None for v in limits.values()):
                # Unlimited pool on this scope: nothing is enforced, so nothing is owed.
                # A leaver walks out with their full own limit, which is the confirmed
                # behaviour and the main foot-gun of the sum rule.
                for uid, _ in members:
                    rows = by_scope.get((uid, kind, sid), 0)
                    if rows == 0 and (uid, kind, sid) not in ledger and \
                            carries.get((uid, kind, sid), 0) == 0:
                        continue
                    await _upsert_ledger(db, pool_id, uid, today, scope, charged=0)
                    await _upsert_carry(db, uid, today, scope, carry=-rows)
                continue

            used = sum(
                by_scope.get((uid, kind, sid), 0) + carries.get((uid, kind, sid), 0)
                for uid, _ in members
            )
            delta = used - sum(ledger.get((uid, kind, sid), 0) for uid, _ in members)
            if delta == 0 and not dissolving:
                continue  # nothing closed on this scope; skip the writes entirely

            shares = apportion(delta, [
                MemberShare(
                    user_id=uid,
                    limit=limits[uid],
                    charged=ledger.get((uid, kind, sid), 0),
                    consumed=by_scope.get((uid, kind, sid), 0),
                )
                for uid, _ in members
            ])

            for uid, _ in members:
                charged = ledger.get((uid, kind, sid), 0) + shares.get(uid, 0)
                await _upsert_ledger(db, pool_id, uid, today, scope, charged=charged)
                await _upsert_carry(
                    db, uid, today, scope,
                    carry=charged - by_scope.get((uid, kind, sid), 0),
                )


async def admit_member(db, pool_id: int, user_id: int, username: str) -> None:
    """Write the joining member's opening ledger and carry rows.

    Called after settle_pool() has closed the interval for the existing members and
    after the membership row is inserted, so the joiner is charged nothing for what the
    pool spent before they arrived (§4.1 example B).

        charged_j = min(limit_j, effective_used_j)
        carry_j   = charged_j - rows_j

    The clamp keeps the invariant SUM(charged) == pool_used alive across admission and
    guarantees ``limit_j - effective_used_j >= 0``, so a joiner can never drag a pool
    below zero headroom. It only bites when an admin lowered the user's limit mid-day
    beneath what they had already spent; otherwise it is a no-op and joining costs the
    joiner nothing.
    """
    from contextlib import AsyncExitStack
    from app import time_utils
    from app.request_tracker import request_tracker
    from app.rate_limit import rate_limit_tracker

    today = time_utils.local_today()

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(request_tracker.pause_flush())
        await request_tracker.flush_pending()

        by_scope = await _member_row_counts(db, [(user_id, username)])
        carries = await _load_carries(db, [user_id], today)

        for scope in rate_limit_tracker.settlement_scopes():
            kind, sid = scope
            limit = rate_limit_tracker.member_limit_for_scope(user_id, kind, sid)
            rows = by_scope.get((user_id, kind, sid), 0)
            effective = max(0, rows + carries.get((user_id, kind, sid), 0))

            if limit is None:
                charged = 0          # unlimited on this scope: nothing is owed
            else:
                charged = min(limit, effective)

            if charged == 0 and rows == 0 and carries.get((user_id, kind, sid), 0) == 0:
                continue             # nothing to record; a missing row reads as 0

            await _upsert_ledger(db, pool_id, user_id, today, scope, charged=charged)
            await _upsert_carry(db, user_id, today, scope, carry=charged - rows)


async def clear_member_settlement(db, pool_id: int, user_id: int) -> None:
    """Drop a departing member's ledger rows, leaving their carry in place.

    The carry is what the member walks out with: it survives the membership and follows
    them into the next pool they join. The ledger row is per-membership bookkeeping and
    would otherwise be resurrected if they rejoined the same pool the same day.
    """
    from sqlalchemy import delete
    from app.auth.models import RequestPoolLedger

    await db.execute(
        delete(RequestPoolLedger).where(
            RequestPoolLedger.pool_id == pool_id,
            RequestPoolLedger.user_id == user_id,
        )
    )
    await db.flush()
