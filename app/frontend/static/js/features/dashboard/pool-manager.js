/**
 * Pool Manager — the authenticated user's request pool.
 *
 * A pool shares one daily request quota. Its limit is the sum of its members'
 * limits and its count is the sum of their consumption, so a member has TWO
 * numbers, not one: what they SENT, and what the pool has CHARGED them. Every
 * confusing thing about pooling lives in the gap between those two, so both are
 * on screen at all times as primary elements — never a tooltip, and never
 * revealed for the first time in the leave dialog.
 *
 * This manager answers two questions and gives each its own card, in two places.
 * The Pool tab is WHO IS IN THIS POOL — a roster, the invite box, and the pool's
 * own actions. The Pool usage card is WHAT THE POOL SPENT, broken down by member
 * or by model; it lives in the *Usage* tab behind that tab's scope toggle, next
 * to the caller's own numbers, because the two are read against each other.
 * UserUsageManager only shows and hides that card — every element inside it is
 * still rendered and fetched here.
 *
 * Every number here comes from the server. `if_i_leave_now` in particular is
 * real settlement, computed and rolled back — a leave dialog that disagrees with
 * what actually happens is worse than no dialog at all.
 */

const POOL_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

class PoolManager {
    constructor() {
        this._cache = null;          // GET /auth/pools/me
        this._loadError = false;
        this._me = null;             // the caller's own username, for the "you" tag
        this._busy = false;          // one mutation in flight at a time

        // Usage panel
        this._usage = null;
        this._usageWindow = 'today';
        this._view = 'member';       // member | model
        this._chart = null;
        // { axis, id } while a row is drilled into, null otherwise — the single
        // drilled-in test. `axis` is always the API's word ('user' | 'model'), not
        // the switch's ('member' | 'model'); the rows carry the mapping.
        this._drill = null;

        // Pool tab sub-tabs. _render() replaces #poolContainer wholesale and
        // _wirePooled() rebinds from scratch, so none of this can live in the DOM:
        // which pane is open, what was typed and who has been fetched are the
        // reader's place in the feature, not part of the payload.
        this._poolTab = 'members';   // members | invite
        this._dirQuery = '';         // the search box's text
        this._dir = null;            // GET /auth/pools/directory -> people[]
        this._dirState = 'idle';     // idle | loading | ready | error

        // All three sliding tracks are fluid, and the indicator is positioned in pixels.
        // Nothing else recomputes it, so a resize would leave it behind its button.
        let raf = 0;
        window.addEventListener('resize', () => {
            cancelAnimationFrame(raf);
            raf = requestAnimationFrame(() => this._positionIndicator());
        });
    }

    // ------------------------------------------------------------------ //
    // Lifecycle
    // ------------------------------------------------------------------ //

    async load() {
        if (this._me === null) await this._loadMe();
        await this._fetchAndRender();
    }

    async _loadMe() {
        try {
            const r = await makeAuthenticatedRequest('/auth/me');
            if (r.ok) this._me = (await r.json()).username || '';
        } catch (e) {
            this._me = '';
        }
    }

    async _fetchAndRender() {
        const container = document.getElementById('poolContainer');
        try {
            const r = await makeAuthenticatedRequest('/auth/pools/me');
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            this._cache = await r.json();
            this._loadError = false;
            this._render(this._cache);
        } catch (error) {
            this._loadError = true;
            console.error('[PoolManager] fetch failed:', error);
            const errorHtml = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Couldn't load your pool. Refresh the page to try again.
                    </div>
                `;
            if (container) container.innerHTML = errorHtml;
            // The Usage tab's pool panel is fed by the same fetch, and it is the tab the
            // user is most likely looking at. Without this its skeleton never resolves.
            this._setUsageStatus(errorHtml);
        }
    }

    // The pool panel's one transient slot: a skeleton until /auth/pools/me answers,
    // then either an error or nothing at all. `null` empties and hides it.
    _setUsageStatus(html) {
        const slot = document.getElementById('poolUsageStatus');
        if (!slot) return;
        slot.classList.toggle('skeleton-stack', html === null);
        slot.removeAttribute('aria-busy');
        slot.innerHTML = html === null ? '' : html;
        slot.style.display = html === null ? 'none' : '';
    }

    // ------------------------------------------------------------------ //
    // Render
    // ------------------------------------------------------------------ //

    _render(data) {
        const container = document.getElementById('poolContainer');
        const usageCard = document.getElementById('poolUsageCard');
        const usageStats = document.getElementById('poolUsageStats');
        const usageEmpty = document.getElementById('poolUsageEmpty');
        if (!container) return;

        // Whether the Usage tab's pool panel is on screen at all is the scope toggle's
        // business (UserUsageManager.setScope). All this decides is what's inside it.
        this._setUsageStatus(null);

        if (!data.pool) {
            container.innerHTML = this._unpooledHtml(data);
            if (usageCard) usageCard.style.display = 'none';
            // The tiles hide with the card: three zeroes over the create-a-pool form
            // would claim a pool that doesn't exist. The empty state stands in.
            if (usageStats) usageStats.style.display = 'none';
            if (usageEmpty) usageEmpty.style.display = '';
            this._drill = null;
            this._wireUnpooled();
            return;
        }

        container.innerHTML = this._pooledHtml(data);
        if (usageEmpty) usageEmpty.style.display = 'none';
        if (usageCard) usageCard.style.display = '';
        if (usageStats) usageStats.style.display = '';
        this._wirePooled();

        if (!this._usage) {
            this.loadUsage().catch(err => console.error('[PoolManager] usage load failed:', err));
        } else if (this._drill) {
            // Whoever was drilled into may have just left, so re-fetch the scoped view
            // rather than painting the pool-wide rows over it.
            this.drillDown(this._drill.axis, this._drill.id);
        } else {
            // Anything that settles the pool — a join, a leave, a removal — changes
            // who belongs in these rows, so the table has to be redrawn or it keeps
            // listing a member who is no longer here.
            this._renderUsage();
        }
    }

    // ---------------------------------------------------------------- //
    // No pool: invites lead, because someone is waiting on an answer.
    // ---------------------------------------------------------------- //

    _unpooledHtml(data) {
        const invites = (data.incoming_invites || []).map(i => this._inviteHtml(i)).join('');
        return `
            ${invites}
            <div class="pool-empty">
                <h6>You're not in a pool</h6>
                <p>
                    A pool shares one daily request limit across its members, so an idle
                    teammate's unused requests are available to whoever needs them. Your own
                    per-minute limit stays yours.
                </p>
                <form id="poolCreateForm" class="mt-3">
                    <div class="mb-2">
                        <label class="form-label" for="poolCreateName">Pool name</label>
                        <input type="text" class="form-control" id="poolCreateName" maxlength="64"
                               required placeholder="e.g. team-alpha">
                    </div>
                    <div class="mb-3">
                        <label class="form-label" for="poolCreateDesc">Description <span class="text-muted">(optional)</span></label>
                        <input type="text" class="form-control" id="poolCreateDesc" maxlength="256"
                               placeholder="What this pool is for">
                    </div>
                    <button type="submit" class="btn btn-primary btn-sm">
                        <i class="fas fa-plus me-1"></i>Create a pool
                    </button>
                </form>
            </div>
        `;
    }

    _inviteHtml(i) {
        const esc = s => window.UIUtils.escapeHtml(s);
        // Stating the pool's current usage AND that the joiner isn't charged for it is
        // what prevents the likeliest support question.
        const used = Number(i.pool_used || 0).toLocaleString();
        const limit = this._fmt(i.pool_limit);
        const mine = this._fmt(i.counterparty_limit);
        const spent = i.pool_limit === null || i.pool_limit === undefined
            ? `The pool has used ${used} requests today.`
            : `The pool has used ${used} of its ${limit} requests today.`;
        return `
            <div class="pool-invite" data-invite="${i.id}">
                <div class="pool-invite__title">
                    ${esc(i.inviter_username)} invited you to ${esc(i.pool_name)}
                </div>
                <p class="pool-invite__body">
                    ${spent} You won't be charged for any of it — you're only charged for what
                    the pool spends after you join.
                    ${i.counterparty_limit === null || i.counterparty_limit === undefined
                        ? 'You have no daily limit, so joining makes this pool unlimited for everyone in it.'
                        : `Your <strong>${mine}</strong> daily requests join the pool.`}
                </p>
                <div class="pool-invite__acts">
                    <button class="btn btn-secondary btn-sm" data-act="decline" data-invite="${i.id}">Decline</button>
                    <button class="btn btn-primary btn-sm" data-act="accept" data-invite="${i.id}">Join pool</button>
                </div>
            </div>
        `;
    }

    // ---------------------------------------------------------------- //
    // Pooled
    // ---------------------------------------------------------------- //

    _pooledHtml(data) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const count = (data.members || []).length;

        // Pending invites get no notice. The Invite tab lists every one of them by name
        // with the Cancel button attached, so a banner would announce a fact the reader
        // is one click from seeing in the place they can act on it.
        const notices = [
            this._unlimitedNotice(data),
            this._inactiveNotice(data),
        ].filter(Boolean).join('');

        return `
            ${notices}
            <div class="pool-head">
                <div class="pool-ident">
                    <span class="pool-socket"></span>
                    <span class="pool-heading">
                        <span class="pool-name">${esc(data.pool.name)}</span>
                        <span class="pool-eyebrow">
                            ${count} ${count === 1 ? 'member' : 'members'} ·
                            owned by ${esc(data.owner_username || '')}
                        </span>
                        ${data.description ? `<span class="pool-desc">${esc(data.description)}</span>` : ''}
                    </span>
                </div>
                ${this._poolMenuHtml(data)}
            </div>
            <div class="pool-block">
                ${this._panelBarHtml(data)}
                <div id="poolPaneMembers"${this._poolTab === 'members' ? '' : ' hidden'}>
                    ${this._membersHtml(data)}
                </div>
                <div id="poolPaneInvite"${this._poolTab === 'invite' ? '' : ' hidden'}>
                    ${this._directoryHtml(data)}
                </div>
            </div>
        `;
    }

    // Two lists of people, not two lenses on one list, so they get real panes. The
    // track itself is the usage card's switch unchanged — one control language per
    // card — and the active button paints its own background because the indicator
    // is positioned in pixels and measures 0 until the Pool tab is visible.
    //
    // "N of 25" replaces the old "this pool is full" footer: the same fact, carried
    // continuously instead of appearing only at the boundary.
    _panelBarHtml(data) {
        const count = (data.members || []).length;
        const max = data.max_members || 25;
        const tab = (key, label) => `
            <button type="button" class="pool-view-btn${this._poolTab === key ? ' is-active' : ''}"
                    data-act="pooltab" data-pooltab="${key}"
                    aria-pressed="${this._poolTab === key}">${label}</button>
        `;
        return `
            <div class="pool-panelbar">
                <div class="pool-view-switch" id="poolTabSwitch" role="group" aria-label="Pool view">
                    <div class="pool-view-switch__indicator" aria-hidden="true"></div>
                    ${tab('members', 'Members')}
                    ${tab('invite', 'Invite')}
                </div>
                <span class="pool-eyebrow">${count} of ${max}</span>
            </div>
        `;
    }

    // Rename / Leave / Delete used to sit in a bar below the whole page. They act on
    // the pool as a whole, so they belong beside its name.
    _poolMenuHtml(data) {
        const ownerItems = data.is_owner ? `
            <li><button class="dropdown-item" type="button" data-act="rename">
                <i class="fas fa-pen me-2"></i>Rename pool
            </button></li>
        ` : '';
        const dangerItems = data.is_owner ? `
            <li><hr class="dropdown-divider"></li>
            <li><button class="dropdown-item dropdown-item-danger" type="button" data-act="dissolve">
                <i class="fas fa-trash me-2"></i>Delete pool
            </button></li>
        ` : '';
        return `
            <div class="dropdown pool-menu">
                <button class="pool-menu__btn" type="button" data-bs-toggle="dropdown"
                        aria-expanded="false" aria-label="Pool actions">
                    <i class="fas fa-ellipsis"></i>
                </button>
                <ul class="dropdown-menu dropdown-menu-end">
                    ${ownerItems}
                    <li><button class="dropdown-item" type="button" data-act="leave">
                        <i class="fas fa-right-from-bracket me-2"></i>Leave pool
                    </button></li>
                    ${dangerItems}
                </ul>
            </div>
        `;
    }

    // Who is in this pool. Membership used to be inferred from whichever quota table
    // you happened to read, which is also where the kick button lived — once per
    // scope, so a two-member pool offered the same destructive action five times.
    _membersHtml(data) {
        const members = data.members || [];
        return `
            <div class="pool-roster">
                <div class="pool-roster-row is-head">
                    <span class="m-name">Member</span>
                    <span class="r-role">Role</span>
                    <span class="r-joined">Joined</span>
                    <span></span>
                </div>
                ${members.map(m => this._rosterRowHtml(data, m)).join('')}
            </div>
        `;
    }

    _rosterRowHtml(data, m) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const tags = [
            m.username === this._me ? 'you' : '',
            m.is_active ? '' : 'deactivated',
        ].filter(Boolean).map(t => `<span class="m-tag">${t}</span>`).join('');

        // The owner kicks others; everyone leaves through the actions menu. The word is
        // on the button rather than behind a bare ✕ — removing someone from a pool is
        // not a close box, and it takes the same shape as Invite and Cancel one tab over
        // so every row-level action in this card reads the same way.
        const act = (data.is_owner && !m.is_owner)
            ? `<button class="pool-state is-kick" data-act="remove" data-user="${m.user_id}"
                       data-name="${esc(m.username)}"
                       aria-label="Kick ${esc(m.username)} from the pool">Kick</button>`
            : '<span></span>';

        return `
            <div class="pool-roster-row${m.is_active ? '' : ' is-off'}">
                <span class="m-name">${esc(m.username)}${tags}</span>
                <span class="r-role" data-label="role">${m.is_owner ? 'owner' : 'member'}</span>
                <span class="r-joined" data-label="joined" title="${esc(this._joinedFull(m.joined_at))}">${esc(this._joinedShort(m.joined_at))}</span>
                ${act}
            </div>
        `;
    }

    _joinedShort(iso) {
        const d = iso ? new Date(iso) : null;
        if (!d || isNaN(d.getTime())) return '—';
        return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
    }

    _joinedFull(iso) {
        const d = iso ? new Date(iso) : null;
        if (!d || isNaN(d.getTime())) return 'Join date unknown';
        return `Joined ${d.toLocaleString()}`;
    }

    // The invite box used to be a bare field that told you a name was wrong only after
    // you submitted it. This is a list you read instead: everyone is here, each row
    // says where they stand, and the only thing left to decide is who.
    //
    // The search box and the rows are rendered apart on purpose — _renderDirectory()
    // rewrites only #poolDirList, so the caret, the selection and any in-flight IME
    // composition survive every keystroke.
    //
    // #poolDirCount is screen-reader-only: a sighted reader can see the list shorten as
    // they type, so a running total is a label for something already on screen. Someone
    // filtering by ear cannot, which is why the live region stays.
    _directoryHtml(data) {
        const full = (data.members || []).length >= (data.max_members || 25);
        return `
            <div class="pool-invite-pane">
                <div class="pool-search">
                    <label class="sr-only" for="poolDirSearch">Search people</label>
                    <i class="fas fa-magnifying-glass pool-search__icon" aria-hidden="true"></i>
                    <input type="text" id="poolDirSearch" class="pool-search__input"
                           placeholder="Search people" maxlength="64"
                           autocomplete="off" spellcheck="false" aria-controls="poolDirList">
                </div>
                ${full ? `
                    <p class="pool-search__note">This pool is full (${data.max_members} members).
                    Someone has to leave before you can invite anyone else.</p>
                ` : ''}
                <p class="sr-only" id="poolDirCount" role="status" aria-live="polite"></p>
                <div class="pool-dir" id="poolDirList"></div>
            </div>
        `;
    }

    // Contains, not starts-with: typing "test" has to keep both "test321" and
    // "usertest". The whole directory is already in memory, so this is a filter over
    // a few hundred strings rather than a request.
    _filteredDir() {
        const people = this._dir || [];
        const q = this._dirQuery.trim().toLowerCase();
        if (!q) return people;
        return people.filter(p => p.username.toLowerCase().includes(q));
    }

    _renderDirectory() {
        const list = document.getElementById('poolDirList');
        const count = document.getElementById('poolDirCount');
        if (!list) return;

        if (this._dirState === 'idle' || this._dirState === 'loading') {
            list.setAttribute('aria-busy', 'true');
            list.innerHTML = `<div class="skeleton-stack">${
                '<div class="skeleton skeleton-line skeleton-line--lg"></div>'.repeat(5)}</div>`;
            if (count) count.textContent = 'Loading people…';
            return;
        }
        list.removeAttribute('aria-busy');

        if (this._dirState === 'error') {
            list.innerHTML = `
                <p class="pool-dir-empty">Couldn't load the list of people.
                <button type="button" class="pool-dir-retry" data-dir-retry>Try again</button></p>
            `;
            if (count) count.textContent = '';
            return;
        }

        const rows = this._filteredDir();
        if (count) {
            count.textContent = rows.length
                ? `${rows.length} ${rows.length === 1 ? 'person' : 'people'}`
                : 'No matches';
        }

        if (!rows.length) {
            const q = window.UIUtils.escapeHtml(this._dirQuery.trim());
            list.innerHTML = `<p class="pool-dir-empty">${
                q ? `No one here matches “${q}”.` : 'There is no one else on this server yet.'}</p>`;
            return;
        }

        list.innerHTML = rows.map(p => this._dirRowHtml(p)).join('');
    }

    // Every state occupies the same slot at the same size, but a state is only a button
    // when there is something to do about it. Member is a fact, so it is a chip. Invited
    // is a fact you can undo, so it becomes Cancel — on the row that names the person,
    // which is where you were already looking when you decided to withdraw it. Invited
    // stays a chip for the one member who cannot cancel it: the server says who may, and
    // offering a control that 403s is worse than not offering one.
    _dirRowHtml(p) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const name = esc(p.username);
        const you = p.username === this._me ? '<span class="m-tag">you</span>' : '';

        let control;
        if (p.state === 'member') {
            control = '<span class="pool-state is-member">Member</span>';
        } else if (p.state === 'invited' && p.can_cancel) {
            control = `
                <button type="button" class="pool-state is-cancel" data-cancel-invite="${p.invite_id}"
                        data-name="${name}" aria-label="Cancel the invite to ${name}">Cancel</button>
            `;
        } else if (p.state === 'invited') {
            control = '<span class="pool-state is-invited">Invited</span>';
        } else {
            const full = this._poolIsFull();
            control = `
                <button type="button" class="pool-state is-invite" data-invite-user="${name}"
                        ${full ? 'disabled title="This pool is full."' : ''}
                        aria-label="Invite ${name}">Invite</button>
            `;
        }

        return `
            <div class="pool-dir-row">
                <span class="m-name">${name}${you}</span>
                ${control}
            </div>
        `;
    }

    _poolIsFull() {
        const data = this._cache;
        if (!data) return false;
        return (data.members || []).length >= (data.max_members || 25);
    }

    async _loadDirectory() {
        if (this._dirState === 'loading') return;
        this._dirState = 'loading';
        this._renderDirectory();
        try {
            const r = await makeAuthenticatedRequest('/auth/pools/directory');
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            this._dir = (await r.json()).people || [];
            this._dirState = 'ready';
        } catch (error) {
            console.error('[PoolManager] directory fetch failed:', error);
            this._dir = null;
            this._dirState = 'error';
        }
        this._renderDirectory();
    }

    setPoolTab(tab) {
        if (tab !== 'members' && tab !== 'invite') return;
        this._poolTab = tab;

        document.querySelectorAll('#poolTabSwitch .pool-view-btn').forEach(b => {
            const on = b.dataset.pooltab === tab;
            b.classList.toggle('is-active', on);
            b.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        const members = document.getElementById('poolPaneMembers');
        const invite = document.getElementById('poolPaneInvite');
        if (members) members.hidden = tab !== 'members';
        if (invite) invite.hidden = tab !== 'invite';

        requestAnimationFrame(() => this._positionIndicator());
        if (tab === 'invite') this._openDirectory();
    }

    // Paint whatever we have, then fetch if we have nothing. Called on every path that
    // reveals the pane, including a re-render that landed with Invite already open.
    _openDirectory() {
        const input = document.getElementById('poolDirSearch');
        if (input) input.value = this._dirQuery;
        this._renderDirectory();
        if (this._dir === null && this._dirState !== 'loading') this._loadDirectory();
    }

    _unlimitedNotice(data) {
        // The main foot-gun of the sum rule, stated where it bites.
        const overall = (data.scopes || []).find(s => s.scope_kind === 'overall');
        if (!overall || !overall.is_unlimited) return '';
        const free = (data.members || [])
            .filter(m => (m.scopes || []).some(s => s.scope_kind === 'overall' && (s.limit === null || s.limit === undefined)))
            .map(m => window.UIUtils.escapeHtml(m.username));
        if (!free.length) return '';
        const who = free.length === 1 ? `<strong>${free[0]}</strong> has` : `<strong>${free.join('</strong>, <strong>')}</strong> have`;
        return `
            <div class="pool-notice">
                <i class="fas fa-infinity"></i>
                <span>${who} no daily limit, so this pool is unlimited. Leaving restores every
                member's full daily limit.</span>
            </div>
        `;
    }

    _inactiveNotice(data) {
        const off = (data.members || []).filter(m => !m.is_active);
        if (!off.length) return '';
        return off.map(m => {
            const name = window.UIUtils.escapeHtml(m.username);
            const own = (m.scopes || []).find(s => s.scope_kind === 'overall');
            const lim = own ? this._fmt(own.limit) : '∞';
            const act = data.is_owner
                ? `<button class="btn btn-secondary btn-sm pool-notice__act" data-act="remove" data-user="${m.user_id}" data-name="${name}">Kick ${name}</button>`
                : '';
            return `
                <div class="pool-notice">
                    <i class="fas fa-user-slash"></i>
                    <span><strong>${name}</strong> is deactivated. They can't send requests, but
                    their ${lim} still counts toward the pool limit.</span>
                    ${act}
                </div>
            `;
        }).join('');
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    _wireUnpooled() {
        const container = document.getElementById('poolContainer');
        if (!container) return;

        const form = document.getElementById('poolCreateForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this._createPool();
            });
        }

        container.querySelectorAll('[data-act]').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.dataset.invite;
                if (btn.dataset.act === 'accept') this._respondInvite(id, 'accept');
                if (btn.dataset.act === 'decline') this._respondInvite(id, 'decline');
            });
        });
    }

    _wirePooled() {
        const container = document.getElementById('poolContainer');
        if (!container) return;

        container.querySelectorAll('[data-act]').forEach(btn => {
            btn.addEventListener('click', () => {
                switch (btn.dataset.act) {
                    case 'leave':    this._confirmLeave(); break;
                    case 'rename':   this._rename(); break;
                    case 'dissolve': this._dissolve(); break;
                    case 'remove':   this._removeMember(btn.dataset.user, btn.dataset.name); break;
                    case 'pooltab':  this.setPoolTab(btn.dataset.pooltab); break;
                }
            });
        });

        const search = document.getElementById('poolDirSearch');
        if (search) {
            search.value = this._dirQuery;
            // No debounce: the directory is already in memory, so a keystroke costs one
            // filter and one write to the row list alone. The input is never rewritten,
            // so the caret stays where it was.
            search.addEventListener('input', () => {
                this._dirQuery = search.value;
                this._renderDirectory();
            });
            search.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && search.value) {
                    e.preventDefault();
                    search.value = '';
                    this._dirQuery = '';
                    this._renderDirectory();
                }
            });
        }

        // Delegated, unlike everything above: the rows are rewritten on every keystroke,
        // and the loop above runs once per render — bound individually, these buttons
        // would lose their handlers on the first character typed. The list container is
        // the stable thing. A distinct attribute keeps that loop from double-binding them.
        const list = document.getElementById('poolDirList');
        if (list) {
            list.addEventListener('click', (e) => {
                if (e.target.closest('[data-dir-retry]')) { this._loadDirectory(); return; }
                const cancel = e.target.closest('[data-cancel-invite]');
                if (cancel) {
                    this._cancelInviteFromDir(cancel.dataset.cancelInvite, cancel.dataset.name);
                    return;
                }
                const btn = e.target.closest('[data-invite-user]');
                if (btn && !btn.disabled) this._inviteUser(btn.dataset.inviteUser);
            });
        }

        // The panes were rendered from _poolTab already; this measures the track, which
        // reads 0 for as long as the Pool tab itself is hidden.
        requestAnimationFrame(() => this._positionIndicator());
        if (this._poolTab === 'invite') this._openDirectory();
    }

    // ------------------------------------------------------------------ //
    // Mutations
    // ------------------------------------------------------------------ //

    async _post(url, options, successFallback) {
        if (this._busy) return null;
        this._busy = true;
        try {
            const r = await makeAuthenticatedRequest(url, options);
            let body = {};
            try { body = await r.json(); } catch (e) { /* empty body */ }
            if (!r.ok) {
                showAlert(body.detail || successFallback || `Request failed (${r.status})`, 'danger');
                return null;
            }
            if (body.message) showAlert(body.message, 'success');
            return body;
        } catch (error) {
            console.error('[PoolManager]', url, error);
            showAlert('Network error. Try again.', 'danger');
            return null;
        } finally {
            this._busy = false;
        }
    }

    async _reload() {
        this._usage = null;
        // Membership just changed, so every state the directory reported is suspect —
        // someone joined, left or was removed. The query and the open sub-tab are the
        // reader's, not the payload's, and survive.
        this._dir = null;
        this._dirState = 'idle';
        // Membership just changed, so a drilled-into member may be gone. Drop the
        // selection and put the tiles back before the fresh payload lands.
        this._exitDrill({ render: false });
        await this._fetchAndRender();
        // Pooled quotas moved, so the Quotas tab's cached copy is stale.
        if (window.QuotasManager) window.QuotasManager._cache = null;
    }

    async _createPool() {
        const name = (document.getElementById('poolCreateName')?.value || '').trim();
        const description = (document.getElementById('poolCreateDesc')?.value || '').trim();
        if (!name) return;
        const body = await this._post('/auth/pools', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: description || null }),
        });
        if (body) await this._reload();
    }

    // Inviting changes nothing about who is in the pool — nobody has joined — so the
    // row flips in place rather than through a full re-render of the list. That is the
    // difference between keeping your search and being thrown back to the top of it.
    async _inviteUser(username) {
        if (!username) return;
        const body = await this._post('/auth/pools/invites', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username }),
        });

        if (!body) {
            // _post already surfaced the error. A rejection usually means our copy of
            // the directory is stale, so throw it away rather than keep showing a row
            // the server has just disagreed with.
            this._dir = null;
            this._dirState = 'idle';
            await this._loadDirectory();
            return;
        }

        // The row becomes Cancel, not a dead "Invited" chip: whoever just sent an invite
        // is the one likeliest to want it back, and create_invite hands back the id so
        // that flip needs no second request. can_cancel is true by construction here —
        // the sender may always cancel.
        const row = (this._dir || []).find(p => p.username === username);
        if (row) {
            row.state = 'invited';
            row.invite_id = body.invite_id;
            row.can_cancel = true;
        }
        // Nothing outside this pane depends on a pending invite — the roster and the
        // "N of 25" count only move when someone actually joins — so the row is the
        // whole update. Re-rendering the card here would cost a DOM replacement to
        // show the reader the same thing back.
        this._renderDirectory();
        this._focusSearch();
    }

    // The twin of _inviteUser, and deliberately its mirror: the row flips back to
    // Invite in place, so changing your mind costs the same as the invite did and
    // leaves the search exactly where it was.
    async _cancelInviteFromDir(inviteId, name) {
        if (!inviteId) return;
        const ok = await window.UIUtils.showConfirmModal(
            'Cancel invite?',
            `${name} will no longer be able to join this pool from this invite. You can invite them again later.`,
            'warning',
        );
        if (!ok) return;

        const body = await this._post(
            `/auth/pools/invites?invite_id=${encodeURIComponent(inviteId)}`, { method: 'DELETE' },
        );

        if (!body) {
            this._dir = null;
            this._dirState = 'idle';
            await this._loadDirectory();
            return;
        }

        const row = (this._dir || []).find(p => String(p.invite_id) === String(inviteId));
        if (row) {
            row.state = 'invitable';
            delete row.invite_id;
            delete row.can_cancel;
        }
        this._renderDirectory();
        this._focusSearch();
    }

    // The button that was clicked has been replaced by its opposite, so focus would
    // otherwise fall to <body>. Park it in the search box, where the next action starts.
    _focusSearch() {
        const input = document.getElementById('poolDirSearch');
        if (!input) return;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
    }

    async _respondInvite(inviteId, action) {
        const body = await this._post(
            `/auth/pools/invites/${action}?invite_id=${encodeURIComponent(inviteId)}`,
            { method: 'POST' },
        );
        if (body) await this._reload();
    }

    _rename() {
        const nameInput = document.getElementById('poolRenameName');
        const descInput = document.getElementById('poolRenameDesc');
        const form = document.getElementById('poolRenameForm');
        const modalEl = document.getElementById('poolRenameModal');
        if (!nameInput || !descInput || !form || !modalEl) return;

        nameInput.value = this._cache?.pool?.name || '';
        descInput.value = this._cache?.description || '';
        const modal = new bootstrap.Modal(modalEl);

        const onSubmit = async (e) => {
            e.preventDefault();
            const name = nameInput.value.trim();
            if (!name) return;
            form.removeEventListener('submit', onSubmit);
            modal.hide();
            const body = await this._post('/auth/pools/me', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                // An empty string clears the description; null would mean "leave it alone".
                body: JSON.stringify({ name, description: descInput.value.trim() }),
            });
            if (body) await this._reload();
        };
        form.addEventListener('submit', onSubmit);
        modalEl.addEventListener('hidden.bs.modal', () => {
            form.removeEventListener('submit', onSubmit);
        }, { once: true });
        modal.show();
    }

    async _dissolve() {
        const name = this._cache?.pool?.name || 'this pool';
        const ok = await window.UIUtils.showConfirmModal(
            `Delete ${name}?`,
            'Every member is settled first, so each keeps the share of today\'s requests they '
            + 'were charged for and nothing else. The pool and its invites are removed.',
            'danger',
        );
        if (!ok) return;
        const body = await this._post('/auth/pools/me', { method: 'DELETE' });
        if (body) await this._reload();
    }

    async _removeMember(userId, name) {
        const ok = await window.UIUtils.showConfirmModal(
            `Kick ${name}?`,
            `${name} is settled first: they keep the share of today's requests they were charged `
            + 'for, and the pool loses their daily limit.',
            'danger',
        );
        if (!ok) return;
        const body = await this._post(
            `/auth/pools/members?user_id=${encodeURIComponent(userId)}`, { method: 'DELETE' },
        );
        if (body) await this._reload();
    }

    // The most important dialog in the feature: user framing, real numbers, no
    // percentages. Everything shown here came back from a real settlement run.
    _confirmLeave() {
        const data = this._cache;
        if (!data || !data.pool) return;

        const esc = s => window.UIUtils.escapeHtml(s);
        const preview = data.if_i_leave_now || [];
        const title = document.getElementById('poolLeaveTitle');
        const body = document.getElementById('poolLeaveBody');
        if (!title || !body) return;

        title.textContent = `Leave ${data.pool.name}?`;
        body.innerHTML = preview.length ? preview.map(p => {
            const sent = Number(p.sent || 0).toLocaleString();
            const charged = Number(p.charged || 0).toLocaleString();
            const scope = p.scope_kind === 'overall' ? '' : ` on ${esc(p.name)}`;
            const tail = (p.limit === null || p.limit === undefined)
                ? `You have no daily limit${scope}, so leaving restores every member's full limit.`
                : `Leaving now leaves you <strong>${Number(p.remaining || 0).toLocaleString()} of your ${Number(p.limit).toLocaleString()}</strong> requests until midnight.`;
            return `
                <p class="mb-3">
                    You've sent ${sent} requests${scope} and been charged ${charged} — your share of
                    what the pool spent while you were a member. ${tail}
                </p>
            `;
        }).join('') : '<p class="mb-3">You haven\'t used anything from this pool today.</p>';

        const modalEl = document.getElementById('poolLeaveModal');
        const confirmBtn = document.getElementById('poolLeaveConfirm');
        const modal = new bootstrap.Modal(modalEl);

        const onConfirm = async () => {
            confirmBtn.removeEventListener('click', onConfirm);
            modal.hide();
            const res = await this._post('/auth/pools/leave', { method: 'POST' });
            if (res) await this._reload();
        };
        confirmBtn.addEventListener('click', onConfirm);
        modalEl.addEventListener('hidden.bs.modal', () => {
            confirmBtn.removeEventListener('click', onConfirm);
        }, { once: true });
        modal.show();
    }

    // ------------------------------------------------------------------ //
    // Usage panel — what each member really sent. Carries never appear here.
    // ------------------------------------------------------------------ //

    // A drill-down survives a window change: the selection is what the member asked
    // about, the window is only the lens they asked through.
    async setWindow(win) {
        const prev = this._drill;
        this._usageWindow = win;
        this._updateWindowButtons(win);
        // Stay in 'back' mode when a drill-down is about to be re-applied, so the view
        // switch doesn't flash in and straight back out between the two renders.
        this._exitDrill({ render: false, header: !prev });
        const ok = await this.loadUsage({ deferBody: !!prev });
        if (!prev) return;
        // A failed load left `_usage` on the *previous* window, so re-applying the
        // drill-down would scope stale numbers under the new window's button. Drop
        // back to the toggle instead — loadUsage has already painted its own error.
        if (ok) await this.drillDown(prev.axis, prev.id);
        else this._setControlsMode('toggle');
    }

    setView(view) {
        if (this._drill) return;   // the switch is hidden while drilled in
        this._view = view;
        document.querySelectorAll('#poolViewSwitch .pool-view-btn').forEach(b => {
            b.classList.toggle('is-active', b.dataset.view === view);
            b.setAttribute('aria-pressed', b.dataset.view === view ? 'true' : 'false');
        });
        this._positionIndicator();
        this._renderUsage();
    }

    async loadUsage({ deferBody = false } = {}) {
        const container = document.getElementById('poolUsageContainer');
        try {
            const params = new URLSearchParams({ window: this._usageWindow });
            const r = await makeAuthenticatedRequest(`/auth/pools/usage?${params.toString()}`);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            this._usage = await r.json();
            // deferBody: a drill-down render follows immediately, so skip the pool-wide
            // chart, tiles and table rather than flashing them before the scoped ones.
            if (!deferBody) {
                this._renderChart(this._usage.timeseries);
                this._renderUsage();
            }
            requestAnimationFrame(() => this._positionIndicator(this._usageWindow));
            return true;
        } catch (error) {
            console.error('[PoolManager] usage fetch failed:', error);
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Couldn't load pool usage. Refresh the page to try again.
                    </div>
                `;
            }
            return false;
        }
    }

    // The pool-wide render. Drilled-in renders go through drillDown().
    _renderUsage() {
        const container = document.getElementById('poolUsageContainer');
        const data = this._usage;
        if (!container || !data) return;

        const totals = data.totals || {};
        this._restoreTopLevelStats();
        this._setStat('poolTotalRequests', totals.requests);
        this._setStat('poolUniqueMembers', totals.unique_members);
        this._setStat('poolUniqueModels', totals.unique_models);

        if (!(totals.requests || 0)) {
            container.innerHTML = this._emptyUsageHtml('No requests from this pool in this period.');
            return;
        }

        if (this._view === 'model') container.innerHTML = this._modelUsageHtml(data);
        else container.innerHTML = this._memberUsageHtml(data);
        this._bindRowClicks(container);
    }

    // ---------------------------------------------------------------- //
    // Drill-down: one member, or one model, across the same window.
    // ---------------------------------------------------------------- //

    async drillDown(axis, id) {
        let data;
        try {
            const params = new URLSearchParams({ window: this._usageWindow, view: axis, id });
            const r = await makeAuthenticatedRequest(`/auth/pools/usage?${params.toString()}`);
            if (!r.ok) {
                const body = await r.json().catch(() => ({}));
                throw new Error(body.detail || `HTTP ${r.status}`);
            }
            data = await r.json();
        } catch (error) {
            // A 403 here means they left the pool between the click and the fetch. The
            // pool-wide view is the honest fallback — never a Back button over a body
            // that a deferred render never painted.
            console.error('[PoolManager] drill-down failed:', error);
            showAlert(error.message || "Couldn't load that breakdown.", 'danger');
            this._exitDrill();
            return;
        }

        this._drill = { axis, id };
        this._setControlsMode('back');
        this._renderChart(data.timeseries);
        this._renderDrilldownStats(axis, data.breakdown);

        const container = document.getElementById('poolUsageContainer');
        if (!container) return;
        container.innerHTML = axis === 'user'
            ? this._modelBreakdownHtml(data.breakdown, id)
            : this._memberBreakdownHtml(data.breakdown, id);
    }

    back() {
        if (!this._usage) return;
        this._exitDrill();
    }

    // `render: false` for callers about to paint something themselves; `header: false`
    // to hold the controls in back mode across a re-applied drill-down.
    _exitDrill({ render = true, header = true } = {}) {
        this._drill = null;
        if (header) this._setControlsMode('toggle');
        if (!render) {
            this._restoreTopLevelStats();
            return;
        }
        this._renderChart(this._usage?.timeseries);
        this._renderUsage();
        // The switch measured 0 while it was display:none, so re-place its indicator
        // one frame later, once layout has settled.
        requestAnimationFrame(() => this._positionIndicator());
    }

    _setControlsMode(mode) {
        const sw = document.getElementById('poolViewSwitch');
        const back = document.getElementById('poolUsageBack');
        if (sw) sw.style.display = mode === 'back' ? 'none' : '';
        if (back) back.style.display = mode === 'back' ? '' : 'none';
    }

    _setStat(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = Number(value || 0).toLocaleString();
    }

    // Selecting a member or a model collapses the three tiles into two, scoped to the
    // selection: its own total, plus a count of the opposite axis. The drill-down
    // payload carries no `totals` at all, so both numbers are summed from the breakdown.
    _renderDrilldownStats(axis, breakdown) {
        const rows = Array.isArray(breakdown) ? breakdown : [];
        this._setStat('poolTotalRequests', rows.reduce((s, r) => s + (r.request_count || 0), 0));
        if (axis === 'user') {
            this._swapStatTiles('pool-stat-members', 'pool-stat-models',
                'poolUniqueModels', rows.length, 'Models Used');
        } else {
            this._swapStatTiles('pool-stat-models', 'pool-stat-members',
                'poolUniqueMembers', rows.length, 'Members');
        }
    }

    _swapStatTiles(hideCardId, keepCardId, keepNumId, value, keepLabel) {
        const hide = document.getElementById(hideCardId);
        const keep = document.getElementById(keepCardId);
        if (hide) hide.style.display = 'none';
        if (!keep) return;
        keep.style.display = '';
        this._setStat(keepNumId, value);
        const label = keep.querySelector('.stat-label');
        if (label) label.textContent = keepLabel;
    }

    _restoreTopLevelStats() {
        const restore = (cardId, text) => {
            const card = document.getElementById(cardId);
            if (!card) return;
            card.style.display = '';
            const label = card.querySelector('.stat-label');
            if (label) label.textContent = text;
        };
        restore('pool-stat-members', 'Members');
        restore('pool-stat-models', 'Models Used');
    }

    _bindRowClicks(container) {
        container.querySelectorAll('.usage-drilldown-row').forEach(row => {
            const open = () => this.drillDown(row.dataset.axis, row.dataset.id);
            row.addEventListener('click', open);
            // A table row is not a button, so Enter and Space are wired by hand.
            row.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
            });
        });
    }

    _emptyUsageHtml(message) {
        return `
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fas fa-chart-bar"></i></div>
                <h3>No Usage Data</h3>
                <p>${window.UIUtils.escapeHtml(message)}</p>
            </div>
        `;
    }

    // The same table the admin Usage tab renders. `rows` is pre-built <tr> markup;
    // every column after the name is numeric, so every one of them right-aligns.
    _usageTable(headers, rows) {
        const head = headers.map((h, i) =>
            `<th${i ? ' class="text-end"' : ''}>${h}</th>`).join('');
        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr>${head}</tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    _pct(n, total) {
        return total > 0 ? ((n / total) * 100).toFixed(1) : '0.0';
    }

    // Every pool-wide row stands for one member or one model, so every one of them
    // opens that breakdown. data-axis carries the API's word, not the switch's.
    _memberUsageHtml(data) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const total = data.totals?.requests || 0;
        const rows = (data.per_member || []).map(m => `
            <tr class="usage-drilldown-row" tabindex="0" style="cursor:pointer;"
                data-axis="user" data-id="${esc(m.user_identity)}"
                title="See which models ${esc(m.user_identity)} used">
                <td>${esc(m.user_identity)}</td>
                <td class="text-end">${Number(m.request_count).toLocaleString()}</td>
                <td class="text-end">${this._pct(m.request_count, total)}%</td>
            </tr>
        `).join('');
        return this._usageTable(['Member', 'Requests', 'Share'], rows);
    }

    _modelUsageHtml(data) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const total = data.totals?.requests || 0;
        const rows = (data.per_model || []).map(m => `
            <tr class="usage-drilldown-row" tabindex="0" style="cursor:pointer;"
                data-axis="model" data-id="${esc(m.model)}"
                title="See who used ${esc(m.model)}">
                <td>${esc(m.model)}</td>
                <td class="text-end">${Number(m.request_count).toLocaleString()}</td>
                <td class="text-end">${this._pct(m.request_count, total)}%</td>
            </tr>
        `).join('');
        return this._usageTable(['Model', 'Requests', 'Share'], rows);
    }

    // The header names what you drilled into; the rows are the other axis. Shares run
    // against this breakdown's own sum — the denominator actually on screen — because
    // the drill-down payload has no pool-wide total to divide by.
    _drilldownHeadHtml(kind, name) {
        const esc = s => window.UIUtils.escapeHtml(s);
        return `<div class="fw-semibold mb-3">${esc(kind)}: ${esc(name)}</div>`;
    }

    // Drilled into a member: which models they used.
    _modelBreakdownHtml(breakdown, id) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const list = Array.isArray(breakdown) ? breakdown : [];
        const head = this._drilldownHeadHtml('Member', id);
        if (!list.length) {
            return head + this._emptyUsageHtml('No requests from this member in this period.');
        }
        const total = list.reduce((s, r) => s + (r.request_count || 0), 0);
        const rows = list.map(r => `
            <tr>
                <td>${esc(r.model)}</td>
                <td class="text-end">${Number(r.request_count).toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}%</td>
            </tr>
        `).join('');
        return head + this._usageTable(['Model', 'Requests', 'Share'], rows);
    }

    // Drilled into a model: who in the pool sent it. No type badge — unlike the admin
    // endpoint, this breakdown carries no user_type.
    _memberBreakdownHtml(breakdown, id) {
        const esc = s => window.UIUtils.escapeHtml(s);
        const list = Array.isArray(breakdown) ? breakdown : [];
        const head = this._drilldownHeadHtml('Model', id);
        if (!list.length) {
            return head + this._emptyUsageHtml('No requests to this model in this period.');
        }
        const total = list.reduce((s, r) => s + (r.request_count || 0), 0);
        const rows = list.map(r => `
            <tr>
                <td>${esc(r.user_identity)}</td>
                <td class="text-end">${Number(r.request_count).toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}%</td>
            </tr>
        `).join('');
        return head + this._usageTable(['Member', 'Requests', 'Share'], rows);
    }

    // The server's bucket labels change shape with the window: `YYYY-MM-DD HH:00`
    // hourly, `YYYY-MM-DD` daily, `YYYY-MM` monthly. Printing the full stamp on every
    // hourly tick is twelve copies of one fact, so an hourly axis shows the time and
    // names the day only where it actually turns over — which `24h` does mid-axis.
    _tickLabel(raw) {
        const s = String(raw ?? '');
        const mon = n => POOL_MONTHS[n - 1] || '';

        const hourly = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/.exec(s);
        if (hourly) {
            const name = mon(Number(hourly[2]));
            return hourly[4] === '00' && name
                ? `${name} ${Number(hourly[3])}`
                : `${hourly[4]}:${hourly[5]}`;
        }

        const daily = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
        if (daily) {
            const name = mon(Number(daily[2]));
            return name ? `${name} ${Number(daily[3])}` : s;
        }

        const monthly = /^(\d{4})-(\d{2})$/.exec(s);
        if (monthly) {
            const name = mon(Number(monthly[2]));
            return name ? `${name} ${monthly[1]}` : s;
        }

        return s;
    }

    _renderChart(timeseries) {
        const canvas = document.getElementById('pool-usage-chart');
        if (!canvas || typeof Chart === 'undefined') return;

        const series = Array.isArray(timeseries) ? timeseries : [];
        const labels = series.map(b => b.label);
        const values = series.map(b => b.count);

        const css = getComputedStyle(document.documentElement);
        const bar = (css.getPropertyValue('--mono-text-primary') || '#fafafa').trim();
        const muted = (css.getPropertyValue('--mono-text-muted') || '#888').trim();
        const grid = (css.getPropertyValue('--border-color') || 'rgba(255,255,255,0.08)').trim();
        const font = (css.getPropertyValue('--font-family-mono') || 'monospace').trim();

        // The update path below re-applies neither `options` nor `type`, so a chart of
        // the wrong type has to be rebuilt rather than mutated.
        if (this._chart && this._chart.config.type !== 'bar') {
            this._chart.destroy();
            this._chart = null;
        }

        if (this._chart) {
            this._chart.data.labels = labels;
            this._chart.data.datasets[0].data = values;
            this._chart.update();
            return;
        }

        // The tick callback is installed once, at construction, and reads the window
        // off the manager each time it runs — the update path above never re-applies
        // `options`, so a callback that closed over the current window would freeze.
        const self = this;

        this._chart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Requests',
                    data: values,
                    backgroundColor: bar,
                    borderWidth: 0,
                    borderRadius: 2,
                    maxBarThickness: 48,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 200 },
                plugins: {
                    legend: { display: false },
                    tooltip: { titleFont: { family: font }, bodyFont: { family: font } },
                },
                scales: {
                    x: {
                        ticks: {
                            color: muted, font: { family: font, size: 10 },
                            maxRotation: 0, autoSkip: true, autoSkipPadding: 12,
                            callback(value) {
                                return self._tickLabel(this.getLabelForValue(value));
                            },
                        },
                        grid: { display: false },
                        border: { display: false },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: muted, font: { family: font, size: 10 }, precision: 0 },
                        grid: { color: grid, drawTicks: false },
                        border: { display: false },
                    },
                },
            },
        });
    }

    _updateWindowButtons(activeKey) {
        const container = document.getElementById('pool-window-container');
        if (!container) return;
        container.querySelectorAll('.time-window__preset').forEach(b => {
            const on = b.dataset.window === activeKey;
            b.classList.toggle('is-active', on);
            b.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        this._positionIndicator(activeKey);
    }

    // The card has three sliding tracks now, and they desync for the same reason: a
    // hidden tab measures 0, so this runs again on tab open, after each load, and on
    // resize. Callers that know the window pass it; everyone else gets current state.
    _positionIndicator(activeKey) {
        this._placeIndicator('pool-window-btns', 'window', activeKey ?? this._usageWindow);
        this._placeIndicator('poolViewSwitch', 'view', this._view);
        // The sub-tab track is the worst case of the three: it sits inside the Pool tab
        // itself, so it is unmeasurable until that tab is opened.
        this._placeIndicator('poolTabSwitch', 'pooltab', this._poolTab);
    }

    _placeIndicator(trackId, attr, activeKey) {
        const track = document.getElementById(trackId);
        const active = track?.querySelector(`[data-${attr}="${activeKey}"]`);
        if (!track || !active || !active.offsetWidth) return;
        track.style.setProperty('--tw-x', `${active.offsetLeft}px`);
        track.style.setProperty('--tw-w', `${active.offsetWidth}px`);
    }

    _fmt(value) {
        return value === null || value === undefined ? '∞' : Number(value).toLocaleString();
    }
}

window.PoolManager = new PoolManager();

// Lazy-load when the Pool tab is opened (the monkey-patch pattern the other
// dashboard managers use — ui-utils.js wires only the admin managers). Only the
// roster is here now; the usage panel is opened by the Usage tab's scope toggle,
// which re-measures its indicators itself.
document.addEventListener('DOMContentLoaded', function () {
    const originalShowTab = window.showTab;
    window.showTab = function (tabName) {
        originalShowTab(tabName);
        if (tabName === 'pool') {
            const mgr = window.PoolManager;
            if (!mgr._cache || mgr._loadError) {
                mgr.load().catch(err => console.error('[PoolManager] Failed to load pool:', err));
            }
            // Every visit, not just the first: the sub-tab track measured 0 while
            // #pool-tab was display:none, and a warm cache skips the load above, so
            // nothing else would ever re-place it on a revisit.
            requestAnimationFrame(() => mgr._positionIndicator());
        }
    };
});
