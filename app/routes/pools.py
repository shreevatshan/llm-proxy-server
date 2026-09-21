"""Request-pool endpoints: forming a pool, moving between pools, and seeing inside one.

A pool shares one daily request quota across its members. Everything about *charging*
lives in app/auth/pools.py; this module owns membership, invitations and visibility.

Two rules shape every handler here:

* **Every composition change settles first.** Joining, leaving, being removed and
  dissolving all call settle_pool() inside the same transaction, *before* the membership
  row moves, so the interval that just closed is charged to exactly the members who were
  present for it. Skipping it anywhere would let a heavy consumer walk away from what the
  pool spent on their behalf.
* **Usage views never apply carries.** GET /auth/pools/usage reports what each member
  actually sent. Only the quota numbers reflect settlement, so attribution and history
  stay exactly as recorded.

Members can read each other's usage, which is the point of pooling, so the membership
check in build_pool_usage() is a security boundary and is tested as one.
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, delete as sa_delete, or_, select, update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.admin import AdminUser
from app.auth.database import get_db, get_user_by_username
from app.auth.middleware import get_current_active_user
from app.auth.models import (
    MAX_POOL_MEMBERS,
    AdminPoolResponse, MyPoolResponse, PoolCreate, PoolInvitationResponse,
    PoolInviteCreate, PoolLeavePreview, PoolMemberResponse, PoolMemberScope,
    PoolMembershipInterval, PoolScopeResponse, PoolSummary, PoolUpdate,
    RequestPool, RequestPoolInvitation, RequestPoolLedger, RequestPoolMember,
    User, UserRpdCarry,
)
from app.auth import pools as pool_settlement
from app import time_utils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/pools", tags=["request-pools"])

POOL_NAME_MAX_LENGTH = 64
POOL_DESCRIPTION_MAX_LENGTH = 256


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Membership intervals
#
# RequestPoolMember says who is in the pool now; PoolMembershipInterval says who was
# in it when, which is what pool *usage* needs -- see the model's docstring. The three
# helpers below are the only writers, so the day-boundary rule (both ends inclusive,
# in local usage dates) lives in exactly one place.
# --------------------------------------------------------------------------- #


async def _open_interval(db: AsyncSession, pool_id: int, user_id: int) -> None:
    """Start a stint today. Called wherever a RequestPoolMember row is created."""
    db.add(PoolMembershipInterval(
        pool_id=pool_id, user_id=user_id, joined_on=time_utils.local_today(),
    ))
    await db.flush()


async def _close_interval(db: AsyncSession, pool_id: int, user_id: int) -> None:
    """End this member's open stint today, inclusive.

    Idempotent via `left_on IS NULL`: closing twice is a no-op, and a member who
    rejoins the same day simply gets a second interval rather than reopening this one.
    """
    await db.execute(
        sa_update(PoolMembershipInterval)
        .where(
            PoolMembershipInterval.pool_id == pool_id,
            PoolMembershipInterval.user_id == user_id,
            PoolMembershipInterval.left_on.is_(None),
        )
        .values(left_on=time_utils.local_today())
    )
    await db.flush()


async def _close_all_intervals(db: AsyncSession, pool_id: int) -> None:
    """End every open stint in a pool that is about to be deleted.

    The rows go away with the pool via ON DELETE CASCADE, so this is only meaningful
    for the transaction's own reads before the delete lands.
    """
    await db.execute(
        sa_update(PoolMembershipInterval)
        .where(
            PoolMembershipInterval.pool_id == pool_id,
            PoolMembershipInterval.left_on.is_(None),
        )
        .values(left_on=time_utils.local_today())
    )
    await db.flush()


def window_bounds(
    window: str, year: Optional[int] = None, month: Optional[int] = None,
) -> Tuple[Optional[date], Optional[date]]:
    """The inclusive local-date range a usage window covers, for clipping spans.

    Mirrors the window handling in get_usage_aggregates / get_usage_timeseries. Only
    used to trim membership stints, so "all" is unbounded on both ends and a stint that
    starts before the window simply starts at the window instead.
    """
    from datetime import timedelta

    today = time_utils.local_today()
    if window == "24h":
        return (time_utils.local_now() - timedelta(hours=24)).date(), today
    if window == "today":
        return today, today
    if window == "yesterday":
        y = today - timedelta(days=1)
        return y, y
    if window == "7d":
        return today - timedelta(days=6), today
    if window == "month" and year and month:
        first = date(year, month, 1)
        nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        return first, nxt - timedelta(days=1)
    if window == "all":
        return None, None
    return today - timedelta(days=29), today      # default 30d


async def member_spans(
    db: AsyncSession,
    pool_id: int,
    *,
    lo: Optional[date] = None,
    hi: Optional[date] = None,
) -> List[Tuple[str, date, date]]:
    """[(username, first_day, last_day)] for every stint overlapping [lo, hi].

    One tuple per stint, so a user who left and rejoined yields two and the gap between
    them is excluded. An open stint ends today. `lo`/`hi` clip each span to the caller's
    window; omitting one leaves that end of the span unclipped.

    Usernames, not ids, because the usage tables are keyed by identity string -- the
    resolution happens here so a rename cannot orphan the interval rows themselves.
    """
    today = time_utils.local_today()
    rows = (await db.execute(
        select(
            User.username,
            PoolMembershipInterval.joined_on,
            PoolMembershipInterval.left_on,
        )
        .join(User, User.id == PoolMembershipInterval.user_id)
        .where(PoolMembershipInterval.pool_id == pool_id)
    )).all()

    spans: List[Tuple[str, date, date]] = []
    for username, joined_on, left_on in rows:
        start = joined_on
        end = left_on or today
        if lo is not None and start < lo:
            start = lo
        if hi is not None and end > hi:
            end = hi
        if start > end:
            continue        # the stint falls entirely outside the window
        spans.append((username, start, end))
    return spans


def _require_user(current_user) -> User:
    """Reject the config-based admin, which has no integer id and is never pooled."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The admin account is exempt from rate limits and cannot join a pool",
        )
    return current_user


async def _pool_of(db: AsyncSession, user_id: int) -> Optional[RequestPoolMember]:
    return (await db.execute(
        select(RequestPoolMember).where(RequestPoolMember.user_id == user_id)
    )).scalar_one_or_none()


async def _get_pool(db: AsyncSession, pool_id: int) -> RequestPool:
    pool = (await db.execute(
        select(RequestPool).where(RequestPool.id == pool_id)
    )).scalar_one_or_none()
    if pool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pool not found")
    return pool


async def _member_rows(db: AsyncSession, pool_id: int) -> List[RequestPoolMember]:
    return list((await db.execute(
        select(RequestPoolMember)
        .where(RequestPoolMember.pool_id == pool_id)
        .order_by(RequestPoolMember.joined_at, RequestPoolMember.id)
    )).scalars().all())


async def _users_by_id(db: AsyncSession, user_ids: List[int]) -> dict:
    if not user_ids:
        return {}
    rows = (await db.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
    return {u.id: u for u in rows}


def _scope_name(scope_kind: str, scope_id: int) -> str:
    """Human label for a scope, from the rate-limit snapshot's group names."""
    from app.rate_limit import rate_limit_tracker

    if scope_kind == "overall":
        return "Overall"
    snapshot = (
        rate_limit_tracker._instance_groups if scope_kind == "instance_group"
        else rate_limit_tracker._groups
    )
    group = snapshot.get(scope_id)
    return getattr(group, "name", None) or f"Group {scope_id}"


async def _invalidate(pool_id: Optional[int], usernames: List[str]) -> None:
    """Drop every cache entry a composition change invalidates, then re-snapshot.

    The leaver falls back to their own scope key, so their username is invalidated too;
    without it they would keep reading the pool's cached count for the length of the TTL.
    """
    from app.rate_limit import rate_limit_tracker

    if pool_id is not None:
        rate_limit_tracker.invalidate_pool(pool_id)
    for name in usernames:
        rate_limit_tracker.invalidate_identity(name)
    await rate_limit_tracker.refresh_now()


async def _member_scopes(
    db: AsyncSession, pool_id: int, members: List[tuple], today
) -> dict:
    """(user_id, scope_kind, scope_id) -> (limit, sent, charged, carry) for a whole pool.

    One usage query for every member and scope, so rendering a pool costs the same
    whether it has two members or twenty.
    """
    from app.rate_limit import rate_limit_tracker

    by_scope = await pool_settlement._member_row_counts(db, members)
    ledger = await pool_settlement._load_ledger(db, pool_id, today)
    carries = await pool_settlement._load_carries(db, [uid for uid, _ in members], today)

    out = {}
    for scope_kind, scope_id in rate_limit_tracker.settlement_scopes():
        for uid, _name in members:
            key = (uid, scope_kind, scope_id)
            out[key] = (
                rate_limit_tracker.member_limit_for_scope(uid, scope_kind, scope_id),
                by_scope.get(key, 0),
                ledger.get(key, 0),
                carries.get(key, 0),
            )
    return out


def _relevant_scopes() -> List[tuple]:
    """Every settlement scope: 'overall' first, then each group by name.

    Idle groups are listed too. The Quotas tab shows a member their limit on every
    group unconditionally, so hiding the quiet ones here made a per-user override look
    like it had not taken effect -- the pool's Overall row still reads the overall
    quota, which the override never touched.
    """
    from app.rate_limit import rate_limit_tracker

    groups = [
        (kind, sid) for kind, sid in rate_limit_tracker.settlement_scopes()
        if kind != "overall"
    ]
    groups.sort(key=lambda s: (_scope_name(*s).lower(), s[0], s[1]))
    return [("overall", 0)] + groups


def _active_scopes(detail: dict, members: List[tuple]) -> List[tuple]:
    """'overall', plus only the groups this pool actually spent on.

    For the leave dialog, which writes a paragraph per scope: a line saying a member
    sent nothing on a group they never touched only buries the number they opened the
    dialog to read.
    """
    return [
        (kind, sid) for kind, sid in _relevant_scopes()
        if kind == "overall" or any(
            detail[(uid, kind, sid)][1] or detail[(uid, kind, sid)][2] for uid, _ in members
        )
    ]


def _build_scope_rows(detail: dict, members: List[tuple], scopes: List[tuple]):
    """Return (pool-level scope rows, per-member scope rows keyed by user_id)."""
    pool_scopes: List[PoolScopeResponse] = []
    per_member: dict = {uid: [] for uid, _ in members}

    for scope_kind, scope_id in scopes:
        name = _scope_name(scope_kind, scope_id)
        limits = [detail[(uid, scope_kind, scope_id)][0] for uid, _ in members]
        unlimited = any(v is None for v in limits)
        pool_limit = None if unlimited else sum(limits)
        # What the limiter itself reads: max(0, rows + carries), summed over the pool.
        # The ledger's `charged` is only rewritten on a composition change, so using it
        # here would report the pool as idle until the next join or leave.
        used = max(0, sum(
            detail[(uid, scope_kind, scope_id)][1] + detail[(uid, scope_kind, scope_id)][3]
            for uid, _ in members
        ))

        pool_scopes.append(PoolScopeResponse(
            scope_kind=scope_kind, scope_id=scope_id, name=name,
            limit=pool_limit, used=used,
            remaining=None if pool_limit is None else max(0, pool_limit - used),
            is_unlimited=unlimited,
        ))

        for uid, _ in members:
            limit, sent, charged, carry = detail[(uid, scope_kind, scope_id)]
            per_member[uid].append(PoolMemberScope(
                scope_kind=scope_kind, scope_id=scope_id, name=name,
                limit=limit, sent=sent, charged=charged, carry=carry,
                net_contribution=0 if limit is None else limit - charged,
            ))

    return pool_scopes, per_member


async def _render_pool(db: AsyncSession, pool: RequestPool) -> tuple:
    """Return (members, pool_scopes, per_member_scopes, users_by_id) for one pool."""
    from app import time_utils
    from app.request_tracker import request_tracker

    # build_pool_usage and settlement both flush first; without this the Pool tab would
    # report a member's `sent` up to one flush interval behind the Usage tab beside it.
    await request_tracker.flush_pending()

    rows = await _member_rows(db, pool.id)
    users = await _users_by_id(db, [r.user_id for r in rows])
    members = [(r.user_id, users[r.user_id].username) for r in rows if r.user_id in users]

    detail = await _member_scopes(db, pool.id, members, time_utils.local_today())
    scopes = _relevant_scopes()
    pool_scopes, per_member = _build_scope_rows(detail, members, scopes)

    member_responses = [
        PoolMemberResponse(
            user_id=r.user_id,
            username=users[r.user_id].username,
            is_owner=(r.user_id == pool.owner_user_id),
            is_active=bool(users[r.user_id].is_active),
            joined_at=r.joined_at,
            scopes=per_member.get(r.user_id, []),
        )
        for r in rows if r.user_id in users
    ]
    return member_responses, pool_scopes, detail, users


def _invite_response(
    invite: RequestPoolInvitation, pool: RequestPool, inviter: str, invitee: str,
    member_count: int = 0, pool_limit: Optional[int] = None, pool_used: int = 0,
    counterparty_limit: Optional[int] = None, counterparty_used: int = 0,
) -> PoolInvitationResponse:
    return PoolInvitationResponse(
        id=invite.id, pool_id=invite.pool_id, pool_name=pool.name,
        inviter_username=inviter, invitee_username=invitee,
        status=invite.status, created_at=invite.created_at,
        responded_at=invite.responded_at,
        pool_member_count=member_count, pool_limit=pool_limit, pool_used=pool_used,
        counterparty_limit=counterparty_limit, counterparty_used=counterparty_used,
    )


# --------------------------------------------------------------------------- #
# Pool lifecycle
# --------------------------------------------------------------------------- #


@router.post("")
async def create_pool(
    payload: PoolCreate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a pool and join it. A pool is never memberless."""
    user = _require_user(current_user)

    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pool name is required")
    if len(name) > POOL_NAME_MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pool name must be at most {POOL_NAME_MAX_LENGTH} characters",
        )
    description = (payload.description or "").strip() or None
    if description and len(description) > POOL_DESCRIPTION_MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Description must be at most {POOL_DESCRIPTION_MAX_LENGTH} characters",
        )

    if await _pool_of(db, user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You're already in a pool. Leave it first to create another.",
        )

    existing = (await db.execute(
        select(RequestPool).where(RequestPool.name.ilike(name))
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A pool named '{name}' already exists",
        )

    try:
        pool = RequestPool(name=name, description=description, owner_user_id=user.id)
        db.add(pool)
        await db.flush()
        db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
        await db.flush()
        await _open_interval(db, pool.id, user.id)
        # The creator is the only member, so there is no interval to close; their own
        # usage enters the pool through the admission clamp like any other joiner.
        await pool_settlement.admit_member(db, pool.id, user.id, user.username)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        # Both pre-checks above have a race window, and two UNIQUE constraints can land
        # here: RequestPoolMember.user_id and RequestPool.name. Re-read to say which,
        # so the loser of a name race is not told they are in a pool they never joined.
        if await _pool_of(db, user.id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You're already in a pool. Leave it first to create another.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A pool named '{name}' already exists",
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Pool creation failed for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create pool",
        )

    await _invalidate(pool.id, [user.username])
    return {"message": f"Created {name}", "pool_id": pool.id}


@router.put("/me")
async def update_my_pool(
    payload: PoolUpdate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename or re-describe the pool. Owner only."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool = await _get_pool(db, membership.pool_id)
    if pool.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the pool owner can change its name or description",
        )

    try:
        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pool name is required")
            if len(name) > POOL_NAME_MAX_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Pool name must be at most {POOL_NAME_MAX_LENGTH} characters",
                )
            clash = (await db.execute(
                select(RequestPool).where(RequestPool.name.ilike(name), RequestPool.id != pool.id)
            )).scalar_one_or_none()
            if clash:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A pool named '{name}' already exists",
                )
            pool.name = name
        if payload.description is not None:
            desc = payload.description.strip() or None
            if desc and len(desc) > POOL_DESCRIPTION_MAX_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Description must be at most {POOL_DESCRIPTION_MAX_LENGTH} characters",
                )
            pool.description = desc
        pool.updated_at = datetime.utcnow()
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Pool update failed for pool {pool.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update pool",
        )
    return {"message": f"Updated {pool.name}"}


async def _dissolve(db: AsyncSession, pool: RequestPool) -> List[str]:
    """Settle every member, then delete the pool. Returns the usernames to invalidate."""
    await pool_settlement.settle_pool(db, pool.id, dissolving=True)
    rows = await _member_rows(db, pool.id)
    users = await _users_by_id(db, [r.user_id for r in rows])
    usernames = [users[r.user_id].username for r in rows if r.user_id in users]
    # Members, invitations, ledger rows and membership intervals go with the pool via
    # ON DELETE CASCADE -- a dissolved pool has no usage view left to feed.
    # Carries deliberately survive: they are what each member walks out carrying.
    await _close_all_intervals(db, pool.id)
    await db.delete(pool)
    await db.flush()
    return usernames


@router.delete("/me")
async def delete_my_pool(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Dissolve the pool. Owner only. Every member is settled before it disappears."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool = await _get_pool(db, membership.pool_id)
    if pool.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the pool owner can delete it",
        )

    pool_id, pool_name = pool.id, pool.name
    try:
        usernames = await _dissolve(db, pool)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Pool deletion failed for pool {pool_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete pool",
        )

    await _invalidate(pool_id, usernames)
    return {"message": f"Deleted {pool_name}"}


# --------------------------------------------------------------------------- #
# Invitations
# --------------------------------------------------------------------------- #


@router.post("/invites")
async def create_invite(
    payload: PoolInviteCreate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Invite a user by username. Any member can invite; the invitee must accept."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool = await _get_pool(db, membership.pool_id)
    invitee_name = (payload.username or "").strip()
    if not invitee_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required")

    invitee = await get_user_by_username(db, invitee_name)
    if invitee is None or not invitee.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active user named '{invitee_name}'",
        )
    if invitee.id == user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You're already in this pool")

    if await _pool_of(db, invitee.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{invitee.username} is already in a pool. They'll need to leave it first.",
        )

    current_members = await _member_rows(db, pool.id)
    if len(current_members) >= MAX_POOL_MEMBERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This pool is full ({MAX_POOL_MEMBERS} members).",
        )

    existing = (await db.execute(
        select(RequestPoolInvitation).where(
            RequestPoolInvitation.pool_id == pool.id,
            RequestPoolInvitation.invitee_user_id == invitee.id,
            RequestPoolInvitation.status == "pending",
        )
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{invitee.username} already has a pending invite to this pool",
        )

    try:
        invite = RequestPoolInvitation(
            pool_id=pool.id, inviter_user_id=user.id, invitee_user_id=invitee.id,
        )
        db.add(invite)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{invitee.username} already has a pending invite to this pool",
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Pool invite creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send invite",
        )

    try:
        from app.auth.webhook import send_notification_webhook
        await send_notification_webhook("pool_invite", {
            "pool_id": pool.id, "pool_name": pool.name,
            "inviter": user.username, "invitee": invitee.username,
        })
    except Exception as e:
        logger.warning(f"pool_invite webhook failed: {e}")

    # The id comes back so the Invite pane can turn the row it just acted on into a
    # Cancel button without re-reading the directory to learn one number.
    return {"message": f"Invited {invitee.username} to {pool.name}", "invite_id": invite.id}


# Rank drives the Invite pane's grouping: who is already in, then who has been asked,
# then everyone you can still ask. Answering "is this person already in?" by looking is
# the pane's whole job, so the grouping is the server's to state rather than the
# browser's to guess.
_DIRECTORY_RANK = {"member": 0, "invited": 1, "invitable": 2}


@router.get("/directory")
async def list_directory(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Everyone this pool could be looking at, each labelled with where they stand.

    The Invite pane is a list you read before acting, not a box you guess into, so this
    returns the whole directory at once and the browser filters it. That is a deliberate
    reversal of the typeahead it replaces, whose two-character floor existed to stop an
    empty box dumping the directory: any pool member can now enumerate every active
    username. The trade was accepted knowingly -- POST /auth/pools/invites already
    discloses the same facts one name at a time, since its errors distinguish "No active
    user named 'X'" from "X is already in a pool".

    Every exclusion mirrors a rejection create_invite already raises, so a row reading
    `invitable` is one the invite will actually accept:

    * inactive users are absent -- create_invite 404s them;
    * users in a *different* pool are absent -- create_invite 400s them, so their row
      would be a control that always fails;
    * the caller is present, as a `member`. The pane is a picture of the pool, and the
      roster one tab over already shows them.

    An `invited` row additionally carries `invite_id` and `can_cancel`, so the pane can
    withdraw the invite from the row itself instead of sending the reader to a separate
    notice for it.

    Pool-full is not decided here. The client knows members and max_members from
    /auth/pools/me and disables the controls itself.
    """
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool_id = membership.pool_id

    # An `invited` row carries its invite's id so the pane can withdraw it in place,
    # and `can_cancel` so it only offers to when the attempt would succeed. The rule
    # is cancel_invite's own -- sender or pool owner -- asked here rather than
    # discovered by a 403 the reader can do nothing about.
    pool = await _get_pool(db, pool_id)
    invites = {
        inv.invitee_user_id: inv
        for inv in (await db.execute(
            select(RequestPoolInvitation).where(
                RequestPoolInvitation.pool_id == pool_id,
                RequestPoolInvitation.status == "pending",
            )
        )).scalars().all()
    }

    # One left join rather than a membership lookup per user: the pool_id that comes
    # back is either this pool or NULL for someone in no pool at all, and anyone whose
    # row names a different pool is filtered out in SQL.
    rows = (await db.execute(
        select(User.id, User.username, RequestPoolMember.pool_id)
        .outerjoin(RequestPoolMember, RequestPoolMember.user_id == User.id)
        .where(
            User.is_active.is_(True),
            or_(
                RequestPoolMember.pool_id == pool_id,
                RequestPoolMember.pool_id.is_(None),
            ),
        )
    )).all()

    people = []
    for uid, username, member_pool_id in rows:
        person = {"user_id": uid, "username": username}
        invite = invites.get(uid)
        if member_pool_id == pool_id:
            person["state"] = "member"
        elif invite is not None:
            person["state"] = "invited"
            person["invite_id"] = invite.id
            person["can_cancel"] = user.id in (invite.inviter_user_id, pool.owner_user_id)
        else:
            person["state"] = "invitable"
        people.append(person)

    # Sorted here rather than in SQL because the rank depends on the invites, and
    # because SQLite orders bare strings by byte -- which would file every capitalised
    # name above every lowercase one.
    people.sort(key=lambda p: (_DIRECTORY_RANK[p["state"]], p["username"].lower()))

    return {"people": people}


@router.get("/invites", response_model=List[PoolInvitationResponse])
async def list_invites(
    direction: str = Query("incoming", pattern="^(incoming|outgoing)$"),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Pending invites addressed to you, or sent from your pool.

    Each carries the counterparty's current daily usage so both sides decide knowingly.
    """
    user = _require_user(current_user)

    if direction == "incoming":
        q = select(RequestPoolInvitation).where(
            RequestPoolInvitation.invitee_user_id == user.id,
            RequestPoolInvitation.status == "pending",
        )
    else:
        membership = await _pool_of(db, user.id)
        if membership is None:
            return []
        q = select(RequestPoolInvitation).where(
            RequestPoolInvitation.pool_id == membership.pool_id,
            RequestPoolInvitation.status == "pending",
        )

    invites = list((await db.execute(q.order_by(RequestPoolInvitation.created_at.desc()))).scalars().all())
    if not invites:
        return []

    return await _hydrate_invites(db, invites)


async def _hydrate_invites(db: AsyncSession, invites: List[RequestPoolInvitation]):
    """Attach pool identity and both sides' overall daily numbers to each invite."""
    from app.rate_limit import rate_limit_tracker

    pool_ids = {i.pool_id for i in invites}
    pools = {
        p.id: p for p in (await db.execute(
            select(RequestPool).where(RequestPool.id.in_(pool_ids))
        )).scalars().all()
    }
    user_ids = {i.inviter_user_id for i in invites} | {i.invitee_user_id for i in invites}
    users = await _users_by_id(db, list(user_ids))

    out = []
    for invite in invites:
        pool = pools.get(invite.pool_id)
        if pool is None:
            continue
        inviter = users.get(invite.inviter_user_id)
        invitee = users.get(invite.invitee_user_id)
        if inviter is None or invitee is None:
            continue

        _pid, member_count, pooled_limit = rate_limit_tracker.pooled_rpd_limits(
            inviter.id, inviter.username
        )
        pool_status = await rate_limit_tracker.get_user_status(inviter.id, inviter.username)
        invitee_status = await rate_limit_tracker.get_user_status(invitee.id, invitee.username)

        out.append(_invite_response(
            invite, pool, inviter.username, invitee.username,
            member_count=member_count if _pid is not None else len(await _member_rows(db, pool.id)),
            pool_limit=pooled_limit if _pid is not None else pool_status.rpd_limit,
            pool_used=pool_status.rpd_count,
            counterparty_limit=invitee_status.rpd_limit,
            counterparty_used=invitee_status.rpd_count,
        ))
    return out


@router.post("/invites/accept")
async def accept_invite(
    invite_id: int = Query(...),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Join the pool. Closes the existing members' interval first, then admits you.

    You are charged nothing for what the pool spent before you arrived, and your own
    limit and usage so far today enter the pool's sums together.
    """
    user = _require_user(current_user)

    invite = (await db.execute(
        select(RequestPoolInvitation).where(RequestPoolInvitation.id == invite_id)
    )).scalar_one_or_none()
    if invite is None or invite.invitee_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This invite was already {invite.status}",
        )
    if await _pool_of(db, user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You're already in a pool. Leave it first to join another.",
        )

    pool = await _get_pool(db, invite.pool_id)
    existing = await _member_rows(db, pool.id)
    if len(existing) >= MAX_POOL_MEMBERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This pool is full ({MAX_POOL_MEMBERS} members).",
        )

    users = await _users_by_id(db, [r.user_id for r in existing])
    affected = [users[r.user_id].username for r in existing if r.user_id in users]

    try:
        # Close the interval against the members who were actually there, THEN join.
        await pool_settlement.settle_pool(db, pool.id)

        db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
        await db.flush()
        await _open_interval(db, pool.id, user.id)
        await pool_settlement.admit_member(db, pool.id, user.id, user.username)

        invite.status = "accepted"
        invite.responded_at = datetime.utcnow()

        # A stale invite must not later let this user switch pools silently.
        others = (await db.execute(
            select(RequestPoolInvitation).where(
                RequestPoolInvitation.invitee_user_id == user.id,
                RequestPoolInvitation.status == "pending",
                RequestPoolInvitation.id != invite.id,
            )
        )).scalars().all()
        for other in others:
            other.status = "superseded"
            other.responded_at = datetime.utcnow()

        await db.commit()
    except IntegrityError:
        # The user_id UNIQUE constraint is the real one-pool-per-user guarantee; the
        # check above is advisory and loses to a concurrent accept.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You're already in a pool. Leave it first to join another.",
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Accepting pool invite {invite_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join pool",
        )

    await _invalidate(pool.id, affected + [user.username])
    return {"message": f"Joined {pool.name}"}


@router.post("/invites/decline")
async def decline_invite(
    invite_id: int = Query(...),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    user = _require_user(current_user)
    invite = (await db.execute(
        select(RequestPoolInvitation).where(RequestPoolInvitation.id == invite_id)
    )).scalar_one_or_none()
    if invite is None or invite.invitee_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This invite was already {invite.status}",
        )
    invite.status = "declined"
    invite.responded_at = datetime.utcnow()
    await db.commit()
    return {"message": "Declined invite"}


@router.delete("/invites")
async def cancel_invite(
    invite_id: int = Query(...),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Withdraw a pending invite. The inviter or the pool owner may cancel."""
    user = _require_user(current_user)
    invite = (await db.execute(
        select(RequestPoolInvitation).where(RequestPoolInvitation.id == invite_id)
    )).scalar_one_or_none()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")

    pool = await _get_pool(db, invite.pool_id)
    if user.id not in (invite.inviter_user_id, pool.owner_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the sender or the pool owner can cancel this invite",
        )
    if invite.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This invite was already {invite.status}",
        )
    invite.status = "cancelled"
    invite.responded_at = datetime.utcnow()
    await db.commit()
    return {"message": "Cancelled invite"}


# --------------------------------------------------------------------------- #
# Leaving and removal
# --------------------------------------------------------------------------- #


async def _remove_member(db: AsyncSession, pool: RequestPool, user_id: int, username: str) -> List[str]:
    """Settle, then take one member out. Returns every username to invalidate.

    Ownership transfers to the earliest-joined remaining member; a pool that loses its
    last member is deleted, freeing its name for reuse.
    """
    await pool_settlement.settle_pool(db, pool.id)

    rows = await _member_rows(db, pool.id)
    users = await _users_by_id(db, [r.user_id for r in rows])
    affected = [users[r.user_id].username for r in rows if r.user_id in users]

    await pool_settlement.clear_member_settlement(db, pool.id, user_id)
    await db.execute(
        sa_delete(RequestPoolMember).where(
            RequestPoolMember.pool_id == pool.id,
            RequestPoolMember.user_id == user_id,
        )
    )
    # The membership row goes; the stint stays, so what this member spent while they
    # were here remains part of the pool's history.
    await _close_interval(db, pool.id, user_id)
    await db.flush()

    remaining = await _member_rows(db, pool.id)
    if not remaining:
        await _close_all_intervals(db, pool.id)
        await db.delete(pool)
    elif pool.owner_user_id == user_id:
        pool.owner_user_id = remaining[0].user_id
        pool.updated_at = datetime.utcnow()
    await db.flush()

    if username not in affected:
        affected.append(username)
    return affected


@router.post("/leave")
async def leave_pool(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Leave the pool, carrying your settled share and nothing else."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool = await _get_pool(db, membership.pool_id)
    pool_id, pool_name = pool.id, pool.name
    try:
        affected = await _remove_member(db, pool, user.id, user.username)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Leaving pool {pool_id} failed for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to leave pool",
        )

    await _invalidate(pool_id, affected)

    from app.rate_limit import rate_limit_tracker
    after = await rate_limit_tracker.get_user_status(user.id, user.username)
    return {
        "message": f"Left {pool_name}",
        "rpd_limit": after.rpd_limit,
        "rpd_count": after.rpd_count,
        "rpd_remaining": after.rpd_remaining,
    }


@router.delete("/members")
async def remove_member(
    user_id: int = Query(...),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove another member. Owner only; the owner leaves via /leave."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    pool = await _get_pool(db, membership.pool_id)
    if pool.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the pool owner can remove members",
        )
    if user_id == user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use 'Leave pool' to remove yourself",
        )

    target = await _pool_of(db, user_id)
    if target is None or target.pool_id != pool.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That user isn't in your pool")

    users = await _users_by_id(db, [user_id])
    target_name = users[user_id].username if user_id in users else ""

    # Read before the commit expires the instance: touching pool.name afterwards would
    # want a lazy refresh, which is a MissingGreenlet on an async session.
    pool_id = pool.id
    pool_name = pool.name
    try:
        affected = await _remove_member(db, pool, user_id, target_name)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Removing member {user_id} from pool {pool_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove member",
        )

    await _invalidate(pool_id, affected)
    # Worded to match the button that got here: the dashboard's roster says Kick.
    return {"message": f"Kicked {target_name} from {pool_name}"}


# --------------------------------------------------------------------------- #
# Reading the pool
# --------------------------------------------------------------------------- #


async def _leave_preview(db: AsyncSession, pool: RequestPool, user_id: int) -> List[PoolLeavePreview]:
    """What leaving right now would leave the caller with, per tier.

    Runs the real settlement and rolls it back, so the number in the confirm dialog is
    the number the user actually gets. A preview the frontend derived for itself would
    drift from settlement the first time either changed.
    """
    from app import time_utils

    today = time_utils.local_today()
    rows = await _member_rows(db, pool.id)
    users = await _users_by_id(db, [r.user_id for r in rows])
    members = [(r.user_id, users[r.user_id].username) for r in rows if r.user_id in users]

    savepoint = await db.begin_nested()
    try:
        await pool_settlement.settle_pool(db, pool.id)
        detail = await _member_scopes(db, pool.id, members, today)
        scopes = _active_scopes(detail, members)
        preview = []
        for scope_kind, scope_id in scopes:
            limit, sent, charged, _carry = detail[(user_id, scope_kind, scope_id)]
            preview.append(PoolLeavePreview(
                scope_kind=scope_kind, scope_id=scope_id,
                name=_scope_name(scope_kind, scope_id),
                limit=limit, sent=sent, charged=charged,
                remaining=None if limit is None else max(0, limit - charged),
            ))
        return preview
    finally:
        await savepoint.rollback()


@router.get("/me", response_model=MyPoolResponse)
async def get_my_pool(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """The caller's pool, or just their pending invites when they aren't in one.

    Every member row carries both `sent` and `charged`. The gap between them is where
    all the confusion about pooling lives, so it is never hidden behind a click.
    """
    user = _require_user(current_user)

    incoming = list((await db.execute(
        select(RequestPoolInvitation)
        .where(
            RequestPoolInvitation.invitee_user_id == user.id,
            RequestPoolInvitation.status == "pending",
        )
        .order_by(RequestPoolInvitation.created_at.desc())
    )).scalars().all())

    membership = await _pool_of(db, user.id)
    if membership is None:
        return MyPoolResponse(incoming_invites=await _hydrate_invites(db, incoming))

    pool = await _get_pool(db, membership.pool_id)
    member_responses, pool_scopes, _detail, _users = await _render_pool(db, pool)

    outgoing = list((await db.execute(
        select(RequestPoolInvitation)
        .where(
            RequestPoolInvitation.pool_id == pool.id,
            RequestPoolInvitation.status == "pending",
        )
        .order_by(RequestPoolInvitation.created_at.desc())
    )).scalars().all())

    owner_name = next((m.username for m in member_responses if m.is_owner), None)

    return MyPoolResponse(
        pool=PoolSummary(id=pool.id, name=pool.name, member_count=len(member_responses)),
        description=pool.description,
        owner_username=owner_name,
        is_owner=(pool.owner_user_id == user.id),
        members=member_responses,
        scopes=pool_scopes,
        if_i_leave_now=await _leave_preview(db, pool, user.id),
        incoming_invites=await _hydrate_invites(db, incoming),
        outgoing_invites=await _hydrate_invites(db, outgoing),
    )


async def build_pool_usage(
    db: AsyncSession,
    pool_id: int,
    window: str = "30d",
    view: Optional[str] = None,
    target: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """The pool usage payload, shared by the member and admin routes.

    The two routes differ only in how they resolve pool_id; authorization is the same
    for both, because it is the pool that bounds the answer. A drill-down target must be
    a member of `pool_id` whoever is asking — an admin reads other users through the
    admin usage endpoints, not through this one. Keeping one implementation is what
    stops the admin's picture from drifting away from what members actually see.

    Carries are never applied here. This reports what each member really sent; only the
    quota numbers reflect settlement.

    Every number below is bounded by membership *intervals*, not by the current roster:
    a member's traffic counts only for the days they were actually in the pool. Without
    that, a heavy user joining today would drag their whole history in, and a member
    leaving would retroactively erase spending that really was the pool's.
    """
    from app.auth.database import (
        get_usage_by_user_and_model, get_usage_timeseries,
        list_instance_groups, list_model_groups,
    )
    from app.request_tracker import request_tracker

    await request_tracker.flush_pending()

    pool = await _get_pool(db, pool_id)
    rows = await _member_rows(db, pool.id)
    users = await _users_by_id(db, [r.user_id for r in rows])
    identities = [users[r.user_id].username for r in rows if r.user_id in users]

    lo, hi = window_bounds(window, year, month)
    spans = await member_spans(db, pool.id, lo=lo, hi=hi)

    # member_count is deliberately the *current* roster: it answers "how many people
    # share this quota now", which is a fact about the pool, not about the window.
    pool_obj = {"id": pool.id, "name": pool.name, "member_count": len(identities)}

    # Drill-down into one member: the membership check is the security boundary. This is
    # the one place a user reads another user's usage, and it is allowed only inside the
    # pool they share. Anyone with a stint overlapping the window qualifies -- a former
    # member appears in the breakdown, so 403-ing their row would be a dead link.
    if view == "user" and target:
        target_spans = [s for s in spans if s[0] == target]
        if not target_spans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="That user isn't in this pool",
            )
        detail = await get_usage_by_user_and_model(
            db, [], window=window, year=year, month=month, restrict_spans=target_spans,
        )
        by_model: dict = {}
        for r in detail:
            by_model[r["model"]] = by_model.get(r["model"], 0) + r["request_count"]
        return {
            "window": window, "pool": pool_obj, "view": "user", "id": target,
            "breakdown": sorted(
                [{"model": m, "request_count": c} for m, c in by_model.items()],
                key=lambda x: (-x["request_count"], x["model"]),
            ),
            "timeseries": await get_usage_timeseries(
                db, window=window, year=year, month=month, restrict_spans=target_spans,
            ),
        }

    if view == "model" and target:
        return {
            "window": window, "pool": pool_obj, "view": "model", "id": target,
            "breakdown": [
                {"user_identity": r["user_identity"], "request_count": r["request_count"]}
                for r in await get_usage_by_user_and_model(
                    db, [], window=window, year=year, month=month, restrict_spans=spans,
                )
                if r["model"] == target
            ],
            "timeseries": await get_usage_timeseries(
                db, filter_model=target, window=window, year=year, month=month,
                restrict_spans=spans,
            ),
        }

    cross = await get_usage_by_user_and_model(
        db, [], window=window, year=year, month=month, restrict_spans=spans,
    )

    # Every view below is a fold of that one result set.
    per_member_totals: dict = {}
    per_model_totals: dict = {}
    for r in cross:
        per_member_totals[r["user_identity"]] = per_member_totals.get(r["user_identity"], 0) + r["request_count"]
        per_model_totals[r["model"]] = per_model_totals.get(r["model"], 0) + r["request_count"]

    # Map each model to its group, instance group first — the same precedence
    # enforcement resolves in, so the usage tab and the quota tab agree on grouping.
    model_groups = await list_model_groups(db)
    instance_groups = await list_instance_groups(db)
    # Each entry carries the settlement scope it belongs to, so the Pool tab can join
    # this fold to the quota numbers in GET /auth/pools/me on (kind, scope_id) rather
    # than on a display name. Ungrouped traffic gets scope_id 0, which matches no scope
    # -- correctly, since it has no quota of its own.
    model_to_group = {
        m.model_id: (g.name, "model_group", g.id)
        for g in model_groups for m in g.members
    }
    provider_to_group = {
        m.provider_key: (g.name, "instance_group", g.id)
        for g in instance_groups for m in g.members
    }

    def _group_of(model: str):
        if not model:
            return ("Other Models", "other", 0)
        prefix = model.split('/', 1)[0] if '/' in model else model
        return (
            provider_to_group.get(prefix)
            or model_to_group.get(model)
            or ("Other Models", "other", 0)
        )

    groups: dict = {}
    for r in cross:
        name, kind, sid = _group_of(r["model"])
        entry = groups.setdefault((kind, sid), {
            "name": name, "kind": kind, "scope_id": sid, "request_count": 0, "_members": {},
        })
        entry["request_count"] += r["request_count"]
        entry["_members"][r["user_identity"]] = entry["_members"].get(r["user_identity"], 0) + r["request_count"]

    per_group = sorted(
        [
            {
                "name": g["name"], "kind": g["kind"], "scope_id": g["scope_id"],
                "request_count": g["request_count"],
                "per_member": sorted(
                    [{"user_identity": u, "request_count": c} for u, c in g["_members"].items()],
                    key=lambda x: -x["request_count"],
                ),
            }
            for g in groups.values()
        ],
        key=lambda x: -x["request_count"],
    )

    def _sorted(d, key_name):
        return sorted(
            [{key_name: k, "request_count": v} for k, v in d.items()],
            key=lambda x: (-x["request_count"], x[key_name]),
        )

    return {
        "window": window,
        "pool": pool_obj,
        "per_member": _sorted(per_member_totals, "user_identity"),
        "per_model": _sorted(per_model_totals, "model"),
        "per_group": per_group,
        "per_member_per_model": cross,
        "timeseries": await get_usage_timeseries(
            db, window=window, year=year, month=month, restrict_spans=spans,
        ),
        "totals": {
            "requests": sum(per_member_totals.values()),
            # Everyone who was in the pool at some point in the window, so the count
            # matches the number of rows in per_member rather than the live roster.
            "unique_members": len({s[0] for s in spans}),
            "unique_models": len(per_model_totals),
        },
    }


@router.get("/usage")
async def get_pool_usage(
    window: str = Query("30d", pattern="^(24h|today|yesterday|7d|30d|month|all)$"),
    view: Optional[str] = Query(None, pattern="^(user|model|group)$"),
    id: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Raw usage across the caller's pool. Members see each other; that is the point."""
    user = _require_user(current_user)
    membership = await _pool_of(db, user.id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="You're not in a pool")

    try:
        return await build_pool_usage(
            db, membership.pool_id, window=window, view=view, target=id,
            year=year, month=month,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pool usage read failed for pool {membership.pool_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load pool usage",
        )


# --------------------------------------------------------------------------- #
# Admin-facing helpers (routes live in app/routes/admin.py)
# --------------------------------------------------------------------------- #


async def pool_membership_map(db: AsyncSession) -> dict:
    """{pool_id: {"name": str, "members": [username, ...]}} for every pool, in one query.

    Deliberately not admin_list_pools: that renders the quota meters and settlement for
    every pool (a _render_pool each, with a flush of its own), and the By Pool usage
    views need nothing but names. Member order matches _member_rows so a pool reads the
    same here as it does everywhere else.

    The usernames are the join key the whole By Pool feature rests on: membership is
    stored by user_id, but User.username is exactly what user_identity holds in the
    usage tables.
    """
    rows = (await db.execute(
        select(RequestPool.id, RequestPool.name, User.username)
        .select_from(RequestPool)
        .outerjoin(RequestPoolMember, RequestPoolMember.pool_id == RequestPool.id)
        .outerjoin(User, User.id == RequestPoolMember.user_id)
        .order_by(RequestPool.name, RequestPoolMember.joined_at, RequestPoolMember.id)
    )).all()

    out: dict = {}
    for pool_id, name, username in rows:
        entry = out.setdefault(pool_id, {"name": name, "members": []})
        # The outer join yields one NULL-username row for a memberless pool. That should
        # not happen — _remove_member deletes a pool that loses its last member — but
        # such a pool still belongs in the map at zero rather than vanishing from it.
        if username is not None:
            entry["members"].append(username)
    return out


async def resettle_after_usage_purge(db: AsyncSession, usernames: List[str]) -> None:
    """Re-settle every pool the given users belong to, after their usage was deleted.

    Settlement charges a pool's members against the rows in today's usage table, so
    purging those rows leaves request_pool_ledger.charged pointing at traffic that no
    longer exists. Every member's row in their own pool tab then reads charged > sent
    until the next composition change or local midnight.

    settle_pool is the entire fix: with the rows gone, delta goes negative by exactly
    the purged amount and apportion refunds it proportionally, pinning each member at
    headroom == charged so every charged lands back at 0. Today's carries are rewritten
    to match and are deliberately *not* deleted by hand — a carry can be a balance the
    member legitimately brought in from a pool they left earlier today, which
    clear_member_settlement leaves behind on purpose.

    Must be called after drop_buffered_usage: settle_pool flushes the tracker itself, so
    running it first would write buffered counts back into the tables just cleared, and
    drop_buffered_usage only clears the in-memory buffer.
    """
    from app.rate_limit import rate_limit_tracker

    if not usernames:
        return

    user_ids = list((await db.execute(
        select(User.id).where(User.username.in_(usernames))
    )).scalars().all())
    if not user_ids:
        return

    pool_ids = list(dict.fromkeys((await db.execute(
        select(RequestPoolMember.pool_id).where(RequestPoolMember.user_id.in_(user_ids))
    )).scalars().all()))
    if not pool_ids:
        return

    try:
        for pool_id in pool_ids:
            await pool_settlement.settle_pool(db, pool_id)
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    # _invalidate widened to N pools with a single refresh. The refresh is the part that
    # matters: only it re-reads UserRpdCarry into the tracker's snapshot, which the
    # settlement above just rewrote. invalidate_identity alone drops the RPD count cache.
    for pool_id in pool_ids:
        rate_limit_tracker.invalidate_pool(pool_id)
    for name in usernames:
        rate_limit_tracker.invalidate_identity(name)
    await rate_limit_tracker.refresh_now()


async def admin_list_pools(db: AsyncSession) -> List[AdminPoolResponse]:
    """Every pool, rendered exactly as its own members see it.

    Reusing _render_pool is what keeps the admin's picture from drifting from the
    members'. Nothing here is new disclosure: GET /admin/usage already exposes every
    user's usage, and pooling does not change what is recorded.
    """
    pools = list((await db.execute(
        select(RequestPool).order_by(RequestPool.name)
    )).scalars().all())

    out = []
    for pool in pools:
        members, scopes, _detail, _users = await _render_pool(db, pool)
        out.append(AdminPoolResponse(
            id=pool.id, name=pool.name, description=pool.description,
            owner_username=next((m.username for m in members if m.is_owner), None),
            created_at=pool.created_at,
            members=members, scopes=scopes,
        ))
    return out


async def admin_delete_pool(db: AsyncSession, pool_id: int) -> str:
    """Settle every member, then force-delete the pool."""
    pool = await _get_pool(db, pool_id)
    name = pool.name
    try:
        usernames = await _dissolve(db, pool)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await _invalidate(pool_id, usernames)
    return name


async def admin_remove_member(db: AsyncSession, user_id: int) -> str:
    """Settle, then force-remove a user from whatever pool they are in."""
    membership = await _pool_of(db, user_id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That user isn't in a pool")

    pool = await _get_pool(db, membership.pool_id)
    users = await _users_by_id(db, [user_id])
    username = users[user_id].username if user_id in users else ""
    pool_id = pool.id
    try:
        affected = await _remove_member(db, pool, user_id, username)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await _invalidate(pool_id, affected)
    return username


async def settle_before_user_delete(
    db: AsyncSession, user_id: int
) -> Tuple[Optional[int], List[str]]:
    """Close the pool's interval before a permanent delete cascades the membership away.

    The FK cascade drops the membership silently, so without this the remaining members
    would keep the departed user's limit-share as free headroom.

    Returns ``(pool_id, usernames)`` for the caller to hand to
    ``invalidate_after_user_delete`` once the delete has committed. Invalidating here
    would be undone on the spot: this runs inside the caller's transaction, so the
    refresh would read the membership row straight back out of the database.
    """
    membership = await _pool_of(db, user_id)
    if membership is None:
        return None, []
    pool = await _get_pool(db, membership.pool_id)
    pool_id = pool.id  # read before _remove_member, which may delete the pool
    users = await _users_by_id(db, [user_id])
    username = users[user_id].username if user_id in users else ""
    affected = await _remove_member(db, pool, user_id, username)
    return pool_id, affected


async def invalidate_after_user_delete(pool_id: Optional[int], usernames: List[str]) -> None:
    """Drop the departed member from the tracker, once the delete has committed.

    Without this the snapshot keeps the deleted user as a member for the length of the
    refresh interval, which inflates the pool's limit by their share and leaves its
    cached day count stale for the same window.
    """
    if pool_id is None and not usernames:
        return
    await _invalidate(pool_id, usernames)
