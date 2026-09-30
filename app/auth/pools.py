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
from typing import Dict, List, Optional, Sequence, Set, Tuple


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


async def _load_members(db, pool_id: int) -> Tuple[List[Tuple[int, str]], Set[int]]:
    """``([(user_id, username)], {inactive user ids})``, ordered by join time then id.

    Deactivated members are returned in the list and flagged separately rather than
    filtered out. They are still members: the rows they sent while their account worked
    are real and still count against the pool. What they no longer do is donate limit --
    see settle_pool, and RateLimitTracker._pooled_rpd_limit on the enforcement side.
    """
    from sqlalchemy import select
    from app.auth.models import RequestPoolMember, User

    rows = (await db.execute(
        select(RequestPoolMember.user_id, User.username, User.is_active)
        .join(User, User.id == RequestPoolMember.user_id)
        .where(RequestPoolMember.pool_id == pool_id)
        .order_by(RequestPoolMember.joined_at, RequestPoolMember.id)
    )).all()
    return (
        [(r.user_id, r.username) for r in rows],
        {r.user_id for r in rows if not r.is_active},
    )


def _fold_models_into_scopes(usage_rows: List[dict]) -> Dict[Tuple[int, str, int], int]:
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
        uid = row["user_id"]
        count = int(row["request_count"])
        scope = rate_limit_tracker.scope_for_model(row["model"]) or ("overall", 0)
        key = (uid, scope[0], scope[1])
        folded[key] = folded.get(key, 0) + count
    return folded


async def _load_ledger(db, pool_id: int, today) -> Dict[Tuple[int, str, int], int]:
    """(user_id, scope_kind, scope_id) -> charged, for today. A missing row reads as 0.

    Selects columns rather than entities on purpose. The writes below go through Core
    bulk upserts, which the ORM identity map never sees, so loading these as entities
    would leave stale objects behind for anything that reads the table again in the same
    session -- _leave_preview settles and then re-reads.
    """
    from sqlalchemy import select
    from app.auth.models import RequestPoolLedger

    rows = (await db.execute(
        select(
            RequestPoolLedger.user_id,
            RequestPoolLedger.scope_kind,
            RequestPoolLedger.scope_id,
            RequestPoolLedger.charged,
        ).where(
            RequestPoolLedger.pool_id == pool_id,
            RequestPoolLedger.usage_date == today,
        )
    )).all()
    return {(r.user_id, r.scope_kind, r.scope_id): int(r.charged) for r in rows}


async def _load_carries(db, user_ids: Sequence[int], today) -> Dict[Tuple[int, str, int], int]:
    """(user_id, scope_kind, scope_id) -> carry, for today. A missing row reads as 0.

    Columns, not entities -- same reason as _load_ledger.
    """
    from sqlalchemy import select
    from app.auth.models import UserRpdCarry

    if not user_ids:
        return {}
    rows = (await db.execute(
        select(
            UserRpdCarry.user_id,
            UserRpdCarry.scope_kind,
            UserRpdCarry.scope_id,
            UserRpdCarry.carry,
        ).where(
            UserRpdCarry.user_id.in_(list(user_ids)),
            UserRpdCarry.usage_date == today,
        )
    )).all()
    return {(r.user_id, r.scope_kind, r.scope_id): int(r.carry) for r in rows}


# Conflict targets. Each must name its unique constraint's columns exactly, in order --
# uq_pool_ledger and uq_rpd_carry in app.auth.models. Any other target is rejected.
_LEDGER_KEY = ["pool_id", "user_id", "usage_date", "scope_kind", "scope_id"]
_CARRY_KEY = ["user_id", "usage_date", "scope_kind", "scope_id"]


def _replace_upsert(db, table, rows: List[dict], index_elements: List[str], column: str):
    """Build one replace-on-conflict bulk upsert for a settlement table.

    Settlement recomputes the whole value every time rather than incrementing it, so on
    conflict the incoming row simply wins. That is the difference from the usage upserts
    in app.auth.database, which add to what is already there -- incrementing `charged`
    or `carry` would double them on every re-settle and cost the idempotence the whole
    scheme rests on.

    SELECT-then-INSERT would lose a concurrent settle of the same pool: both would read a
    missing row and both would insert. Letting the database resolve the conflict makes the
    write safe without holding a read lock across the apportionment. The per-pool lock
    does not make this redundant -- user_rpd_carries is keyed without pool_id, so two
    *different* pools settling at once contend for the same carry row whenever a user
    moved between them today, which no per-pool lock can serialise.

    Dialect is taken off the bind rather than hardcoded: DATABASE_URL is env-overridable
    and the SQLite-only code elsewhere is each guarded, so pinning it here would be the
    one place pooling silently assumes SQLite.
    """
    name = db.bind.dialect.name if db.bind is not None else "sqlite"
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    elif name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    else:
        raise NotImplementedError(
            f"Pool settlement needs an upsert construct for dialect '{name}'."
        )

    stmt = dialect_insert(table).values(rows)
    return stmt.on_conflict_do_update(
        index_elements=index_elements,
        set_={column: getattr(stmt.excluded, column)},
    )


async def _write_settlement(db, ledger_rows: List[dict], carry_rows: List[dict]) -> None:
    """Write a whole settlement in two statements instead of two per member per scope.

    Flushes rather than commits, so this still composes into the caller's transaction --
    that is what lets _leave_preview run a real settlement inside a savepoint and roll it
    back. Rows are unique on their conflict key by construction (one per member per
    scope); a duplicate would make SQLite refuse to touch the same row twice.
    """
    from app.auth.models import RequestPoolLedger, UserRpdCarry

    if ledger_rows:
        await db.execute(_replace_upsert(
            db, RequestPoolLedger.__table__, ledger_rows, _LEDGER_KEY, "charged"))
    if carry_rows:
        await db.execute(_replace_upsert(
            db, UserRpdCarry.__table__, carry_rows, _CARRY_KEY, "carry"))
    if ledger_rows or carry_rows:
        await db.flush()


async def _member_row_counts(db, members: Sequence[Tuple[int, str]]) -> Dict[Tuple[int, str, int], int]:
    """Today's raw request_usage counts per member per scope, read fresh.

    request_tracker buffers usage in memory and flushes every FLUSH_INTERVAL seconds, so a naive read can
    miss up to a minute of traffic -- and it would miss it in the worst direction, since
    the missing requests are exactly the ones a departing member most recently made.
    The caller holds pause_flush() around this so a concurrent flush cannot land between
    the read and the write and shift the counts underneath the settlement.

    Deliberately NOT filtered by pool_id: a member's charge is what they sent today
    wherever they were, so rows stamped with an earlier pool (or none) count too. That
    is what keeps SUM(charged) == pool_used across joins and leaves.
    """
    from app.auth.database import get_usage_by_user_and_model

    rows = await get_usage_by_user_and_model(
        db, [uid for uid, _ in members], window="today",
    )
    return _fold_models_into_scopes(rows)


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

    members, inactive = await _load_members(db, pool_id)
    if not members:
        return

    today = time_utils.local_today()

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(request_tracker.pause_flush())
        await request_tracker.flush_pending()

        by_scope = await _member_row_counts(db, members)
        ledger = await _load_ledger(db, pool_id, today)
        carries = await _load_carries(db, [uid for uid, _ in members], today)

        ledger_rows: List[dict] = []
        carry_rows: List[dict] = []

        def record(uid: int, scope, charged: int, rows: int) -> None:
            kind, sid = scope
            ledger_rows.append(dict(
                pool_id=pool_id, user_id=uid, usage_date=today,
                scope_kind=kind, scope_id=sid, charged=int(charged),
            ))
            carry = int(charged) - int(rows)
            # A missing carry row already reads as 0; only write one if it says something
            # or if there is already a row to correct.
            if carry != 0 or (uid, kind, sid) in carries:
                carry_rows.append(dict(
                    user_id=uid, usage_date=today,
                    scope_kind=kind, scope_id=sid, carry=carry,
                ))

        for scope in rate_limit_tracker.settlement_scopes():
            kind, sid = scope
            limits = {
                uid: rate_limit_tracker.member_limit_for_scope(uid, kind, sid)
                for uid, _ in members
            }

            # Only active members decide whether the pool is unlimited. A disabled
            # account cannot send anything, so letting one keep the pool uncapped would
            # leave the grant standing with nobody able to withdraw it.
            if any(limits[uid] is None for uid, _ in members if uid not in inactive):
                # Unlimited pool on this scope: no cap is enforced while the pool stands,
                # so there is nothing to apportion -- each member simply owns what they
                # sent, clamped to their own limit:
                #
                #     charged = min(own_limit, effective_used)
                #
                # which is exactly the clamp admit_member applies, so admission and
                # settlement agree on the same inputs. Charging a flat 0 here instead --
                # the old behaviour -- erased the member's day: joining a pool that had
                # any unlimited member and leaving again reset the counter to zero, on
                # demand, and the `delta == 0` skip below then froze that wipe in place
                # for whoever stayed behind after the unlimited member left.
                #
                # The grant survives: nobody is capped while the pool is unlimited. It is
                # only settled honestly on the way out, so a member who spends past their
                # own limit walks out at that limit rather than at zero.
                for uid, _ in members:
                    rows = by_scope.get((uid, kind, sid), 0)
                    carry_in = carries.get((uid, kind, sid), 0)
                    if rows == 0 and (uid, kind, sid) not in ledger and carry_in == 0:
                        continue
                    own = limits[uid]
                    effective = max(0, rows + carry_in)
                    charged = effective if own is None else min(own, effective)
                    record(uid, scope, charged, rows)
                continue

            used = sum(
                by_scope.get((uid, kind, sid), 0) + carries.get((uid, kind, sid), 0)
                for uid, _ in members
            )
            delta = used - sum(ledger.get((uid, kind, sid), 0) for uid, _ in members)
            if delta == 0 and not dissolving:
                continue  # nothing closed on this scope; skip the writes entirely

            # An inactive member's limit is pinned at what they have already been
            # charged, which is zero headroom: they absorb none of a new delta, so the
            # active members split all of it. A refund still reaches them, because
            # refund headroom is `charged` -- if their rows are purged their charge must
            # come down with everyone else's.
            shares = apportion(delta, [
                MemberShare(
                    user_id=uid,
                    limit=(ledger.get((uid, kind, sid), 0) if uid in inactive
                           else limits[uid]),
                    charged=ledger.get((uid, kind, sid), 0),
                    consumed=by_scope.get((uid, kind, sid), 0),
                )
                for uid, _ in members
            ])

            for uid, _ in members:
                charged = ledger.get((uid, kind, sid), 0) + shares.get(uid, 0)
                record(uid, scope, charged, by_scope.get((uid, kind, sid), 0))

        await _write_settlement(db, ledger_rows, carry_rows)


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

        ledger_rows: List[dict] = []
        carry_rows: List[dict] = []

        for scope in rate_limit_tracker.settlement_scopes():
            kind, sid = scope
            limit = rate_limit_tracker.member_limit_for_scope(user_id, kind, sid)
            rows = by_scope.get((user_id, kind, sid), 0)
            carry_in = carries.get((user_id, kind, sid), 0)
            effective = max(0, rows + carry_in)

            # Unlimited on this scope: the joiner owns what they sent, uncapped. Same
            # clamp settle_pool applies, so admission and settlement agree.
            charged = effective if limit is None else min(limit, effective)

            if charged == 0 and rows == 0 and carry_in == 0:
                continue             # nothing to record; a missing row reads as 0

            ledger_rows.append(dict(
                pool_id=pool_id, user_id=user_id, usage_date=today,
                scope_kind=kind, scope_id=sid, charged=int(charged),
            ))
            carry = int(charged) - int(rows)
            if carry != 0 or (user_id, kind, sid) in carries:
                carry_rows.append(dict(
                    user_id=user_id, usage_date=today,
                    scope_kind=kind, scope_id=sid, carry=carry,
                ))

        await _write_settlement(db, ledger_rows, carry_rows)


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
