"""Named monkeypatches applied inside each A/B worker."""
def apply(name):
    import packer, geometry

    if name == 'nosupport':
        packer._footprint_supported = lambda *a, **k: True

    elif name == 'support50':
        def fp(cx, cy, fx, fy, support, inset=0.004, grid=5, bridge=geometry.GAP/2.0+0.003):
            hx = max(fx/2.0-inset, 0.0); hy = max(fy/2.0-inset, 0.0)
            hit = tot = 0
            for i in range(grid):
                px = cx-hx+(2*hx)*i/(grid-1) if grid > 1 else cx
                for j in range(grid):
                    py = cy-hy+(2*hy)*j/(grid-1) if grid > 1 else cy
                    tot += 1
                    if any(abs(px-b['center'][0]) <= b['half'][0]+bridge and
                           abs(py-b['center'][1]) <= b['half'][1]+bridge for b in support):
                        hit += 1
            return hit >= 0.5*tot
        packer._footprint_supported = fp

    elif name == 'desperate':
        # keep the strict support rule, but if the whole search comes up
        # empty, try again with it switched off rather than dying.
        orig = packer.best_placement
        strict = packer._footprint_supported
        def bp(cstate, item_spec, time_deadline=None, relax=False):
            r = orig(cstate, item_spec, time_deadline, relax)
            if r is not None:
                return r
            packer._footprint_supported = lambda *a, **k: True
            try:
                return orig(cstate, item_spec, time_deadline, relax)
            finally:
                packer._footprint_supported = strict
        packer.best_placement = bp

    elif name == 'toptol':
        packer.TOP_TOL = 0.06

    elif name == 'none':
        pass
    else:
        raise SystemExit('unknown patch ' + name)

def _mk_com(frac, need_center=True, grid=7):
    import geometry
    bridge = geometry.GAP/2.0 + 0.003
    def fp(cx, cy, fx, fy, support, inset=0.004, grid=grid, bridge=bridge):
        hx = max(fx/2.0-inset, 0.0); hy = max(fy/2.0-inset, 0.0)
        if need_center and not any(
                abs(cx-b['center'][0]) <= b['half'][0]+bridge and
                abs(cy-b['center'][1]) <= b['half'][1]+bridge for b in support):
            return False
        hit = tot = 0
        sx = sy = 0.0
        for i in range(grid):
            px = cx-hx+(2*hx)*i/(grid-1)
            for j in range(grid):
                py = cy-hy+(2*hy)*j/(grid-1)
                tot += 1
                if any(abs(px-b['center'][0]) <= b['half'][0]+bridge and
                       abs(py-b['center'][1]) <= b['half'][1]+bridge for b in support):
                    hit += 1; sx += px; sy += py
        if hit < frac*tot:
            return False
        # centroid of the contact patch must sit under the item's centre
        cxs, cys = sx/hit, sy/hit
        return abs(cxs-cx) <= hx*0.35 and abs(cys-cy) <= hy*0.35
    return fp

def apply2(name):
    import packer
    if name.startswith('com'):
        frac = float(name[3:]) / 100.0
        packer._footprint_supported = _mk_com(frac)
    else:
        apply(name)

def apply3(name):
    import packer
    if name.startswith('grad'):
        packer._footprint_supported = _mk_com(0.60)
        packer.W_FLAT = 600.0
        import survival
        survival.TOPK = int(name[4:]) if len(name) > 4 else 3
        packer.choose_action = survival.choose_action_graded
    elif name.startswith('surv'):
        # surv[<frac>] on top of com60 + flat
        packer._footprint_supported = _mk_com(0.60)
        packer.W_FLAT = 600.0
        import survival
        if len(name) > 4:
            survival.TOPK = int(name[4:])
        packer.choose_action = survival.choose_action_survival
    else:
        apply2(name)

def _mk_layercap(band):
    """Hard layer discipline: refuse to build above z_lo + k*band until
    nothing fits below it. Targets the measured dominant failure -- at the
    stuck state 77-100% of candidates are rejected because the landing
    height plus the item's own height overruns the ceiling, which happens
    because the skyline is rough (sd 0.53-0.61 m) and any large footprint
    straddles a tall stack."""
    import time, packer
    orig_bp = packer.best_placement
    def bp(cstate, item_spec, time_deadline=None, relax=False):
        g = cstate.geom
        z0, zh = g['z_lo'], g['z_hi']
        cands = packer._collect(cstate, item_spec, relax, packer._xy_candidates)
        k = 1
        while z0 + band * (k - 1) < zh:
            if time_deadline is not None and time.perf_counter() > time_deadline:
                break
            cap = min(z0 + band * k, zh)
            sub = [c for c in cands if c[3] + c[6] <= cap + 1e-9]
            if sub:
                r = packer._first_valid(cstate, sub, time_deadline)
                if r is not None:
                    return r
            k += 1
        return orig_bp(cstate, item_spec, time_deadline, relax)
    packer.best_placement = bp

def apply4(name):
    if name.startswith('layer'):
        _mk_layercap(float(name[5:]) / 100.0)
    else:
        apply3(name)

def apply5(name):
    import packer
    if name.startswith('ro'):
        import poollook
        if len(name) > 2:
            poollook.ROLLOUT_M = int(name[2:])
        packer.choose_action = poollook.choose_action_rollout
    elif name.startswith('rnd'):
        import poollook
        packer.choose_action = poollook.make_random_pool_chooser(int(name[3:]))
    elif name.startswith('d2'):
        import poollook
        parts = name[2:].split('_')
        if parts and parts[0]:
            poollook.M_CANDIDATES = int(parts[0])
        if len(parts) > 1 and parts[1]:
            poollook.R_FOLLOW = int(parts[1])
        packer.choose_action = poollook.choose_action_depth2
    else:
        apply4(name)


# ---------------------------------------------------------------------------
# Heightmap-minimization ("waste") penalty.
#
# The score in packer._collect is purely positional -- low, back, left, flat.
# W_LOW penalises a high resting height, but it cannot tell the two reasons
# for one apart: an item sitting high on a solid stack (no void underneath)
# and an item whose corner clips a tall neighbour so most of its footprint
# bridges over empty air. The second is what makes the skyline rough
# (measured sd 0.53-0.61 m, docs/RESEARCH.md), and a rough skyline is what
# makes 77-100% of later candidates overrun the ceiling.
#
# This adds Wang & Hauser's heightmap-minimization term (ICRA 2019) as a
# penalty: the mean void depth under the item's footprint. Computed from a
# coarse heightmap + summed-area table built once per _collect call, so the
# per-candidate cost is O(1) and fully vectorised.
CELL = 0.03


def _heightmap_sat(cstate):
    import numpy as np
    g = cstate.geom
    x0, x1 = g['x_lo'], g['x_hi']
    y0, y1 = g['y_lo'], g['y_hi']
    nx = max(int((x1 - x0) / CELL) + 1, 1)
    ny = max(int((y1 - y0) / CELL) + 1, 1)
    H = np.full((nx, ny), g['floor_struct_z'], dtype=np.float64)
    for b in cstate.boxes + cstate.static_obstacles:
        cx, cy, cz = b['center']
        hx, hy, hz = b['half']
        i0 = max(int((cx - hx - x0) / CELL), 0)
        i1 = min(int((cx + hx - x0) / CELL) + 1, nx)
        j0 = max(int((cy - hy - y0) / CELL), 0)
        j1 = min(int((cy + hy - y0) / CELL) + 1, ny)
        if i1 > i0 and j1 > j0:
            np.maximum(H[i0:i1, j0:j1], cz + hz, out=H[i0:i1, j0:j1])
    S = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    S[1:, 1:] = H.cumsum(0).cumsum(1)
    return S, nx, ny, x0, y0


def _mk_waste(weight, by_volume=False):
    import numpy as np
    import packer
    orig = packer._collect

    def collect(cstate, item_spec, relax, xy_fn):
        cands = orig(cstate, item_spec, relax, xy_fn)
        if not cands or not (cstate.boxes or cstate.static_obstacles):
            return cands
        S, nx, ny, x0, y0 = _heightmap_sat(cstate)
        cx = np.array([c[1] for c in cands])
        cy = np.array([c[2] for c in cands])
        hx = np.array([c[4] for c in cands])
        hy = np.array([c[5] for c in cands])
        hz = np.array([c[6] for c in cands])
        cz = np.array([c[3] for c in cands])
        bottom = cz - hz
        i0 = np.clip(((cx - hx - x0) / CELL).astype(np.int64), 0, nx - 1)
        i1 = np.clip(((cx + hx - x0) / CELL).astype(np.int64) + 1, 1, nx)
        j0 = np.clip(((cy - hy - y0) / CELL).astype(np.int64), 0, ny - 1)
        j1 = np.clip(((cy + hy - y0) / CELL).astype(np.int64) + 1, 1, ny)
        i1 = np.maximum(i1, i0 + 1)
        j1 = np.maximum(j1, j0 + 1)
        tot = S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]
        cnt = (i1 - i0) * (j1 - j0)
        mean_h = tot / cnt
        waste = np.maximum(bottom - mean_h, 0.0)
        if by_volume:
            waste = waste * (4.0 * hx * hy)
        scores = np.array([c[0] for c in cands]) - weight * waste
        out = [(float(s),) + tuple(c[1:]) for s, c in zip(scores, cands)]
        out.sort(key=lambda c: -c[0])
        return out

    packer._collect = collect


_apply5_prev = apply5


def apply5(name):  # noqa: F811
    if name.startswith('wastev'):
        _mk_waste(float(name[6:]), by_volume=True)
    elif name.startswith('waste'):
        _mk_waste(float(name[5:]))
    else:
        _apply5_prev(name)


# ---------------------------------------------------------------------------
# choose_action budget allocation.
#
# The shipped rule splits the policy budget evenly over every
# (pool item x container) pair. With look_ahead=20 that is 3.5s / 18 =
# 0.19s per best_placement, while one best_placement at a half-full
# container measures 0.25-0.35s (tools/diag_prof.py). Every call then
# times out and choose_action returns None -- the agent dies reporting
# "nothing fits" at a state where a 3cm brute force finds 13543 valid
# placements (tools/diag_gap.py on P00). Giving the same agent a 15s
# budget is worth +2.78 norm on the 32-scene pool, so the starvation is
# real and it is worth about as much as every scoring change combined.
#
# Fix: evaluate the pool biggest-item-first (which is the order the
# W_ITEM_VOL term prefers anyway) and give each call at least MIN_CALL
# seconds, letting the tail of the pool go unevaluated rather than
# evaluating all of it badly.
def _mk_budget(min_call, bigfirst=True, reserve=0.0):
    import time
    import packer
    vol = lambda s: s['length'] * s['width'] * s['height']

    def choose_action(container_states, pool, time_deadline, relax=False,
                      require_support=True):
        any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
        items = sorted(pool, key=lambda t: -vol(t[1])) if bigfirst else list(pool)
        n_calls = max(len(items) * len(container_states), 1)
        main_deadline = time_deadline - reserve
        best = None
        best_score = float('-inf')
        done = 0
        for pool_idx, spec in items:
            for cidx, cstate in container_states.items():
                now = time.perf_counter()
                left = n_calls - done
                done += 1
                if now > main_deadline:
                    continue
                per = max((main_deadline - now) / max(left, 1), min_call)
                r = packer.best_placement(cstate, spec, min(main_deadline, now + per),
                                          relax, require_support)
                if r is None:
                    continue
                pos, orn, score = r
                score += packer.priority_bonus(spec, cstate.geom, any_prio)
                score += packer.W_ITEM_VOL * vol(spec)
                if score > best_score:
                    best_score = score
                    best = (pool_idx, cidx, pos, orn)
        if best is not None or reserve <= 0.0:
            return best
        # Rescue: the even split found nothing anywhere. Spend the reserve on
        # the biggest item alone, undivided, and take the first hit.
        for pool_idx, spec in items:
            for cidx, cstate in container_states.items():
                if time.perf_counter() > time_deadline:
                    return None
                r = packer.best_placement(cstate, spec, time_deadline, relax,
                                          require_support)
                if r is not None:
                    pos, orn, _ = r
                    return (pool_idx, cidx, pos, orn)
        return None

    packer.choose_action = choose_action


_apply5_prev2 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('bud'):
        # bud<min_call_ms>[_r<reserve_ms>][n]  (trailing 'n' = keep pool order)
        body = name[3:]
        bigfirst = True
        if body.endswith('n'):
            bigfirst = False
            body = body[:-1]
        parts = body.split('_r')
        mc = float(parts[0]) / 1000.0
        rs = float(parts[1]) / 1000.0 if len(parts) > 1 else 0.0
        _mk_budget(mc, bigfirst, rs)
    else:
        _apply5_prev2(name)


# ---------------------------------------------------------------------------
# Faster _landing. Semantics are meant to be byte-identical; the win is that
# the boxes are pre-flattened into tuples (the shipped version does six dict
# lookups per box per candidate) and pre-sorted by top height, which lets the
# scan stop as soon as no remaining box can still be a supporter.
#
# _landing is 46% of best_placement's runtime (tools/diag_prof.py: 0.279s of
# 0.600s, 29086 calls for one item), and compute is the binding constraint --
# tripling the policy budget is worth +2.78 norm.
def _mk_fastland():
    import packer
    TOP_TOL = packer.TOP_TOL

    cache = {}

    def index_of(cstate):
        key = id(cstate)
        hit = cache.get(key)
        n = len(cstate.boxes)
        if hit is None or hit[0] != n:
            idx = sorted(((b['center'][2] + b['half'][2], b['center'][0], b['half'][0],
                           b['center'][1], b['half'][1], b)
                          for b in cstate.boxes + cstate.static_obstacles),
                         key=lambda e: -e[0])
            cache.clear()
            cache[key] = (n, idx)
            return idx
        return hit[1]

    def _landing(cstate, cx, cy, hx, hy, inset=0.004):
        ex = hx - inset
        ey = hy - inset
        floor = cstate.geom['floor_struct_z']
        support_top = floor
        supporters = None
        limit = floor - TOP_TOL
        for top, bx, bhx, by, bhy, b in index_of(cstate):
            if top < limit:
                break
            dx = cx - bx
            if dx < 0.0:
                dx = -dx
            if dx >= ex + bhx:
                continue
            dy = cy - by
            if dy < 0.0:
                dy = -dy
            if dy >= ey + bhy:
                continue
            if supporters is None:
                if top > floor:
                    support_top = top
                limit = support_top - TOP_TOL
                supporters = [b]
            else:
                supporters.append(b)
        return support_top, supporters

    packer._landing = _landing


_apply5_prev3 = apply5


def apply5(name):  # noqa: F811
    if name == 'fastland':
        _mk_fastland()
    elif name.startswith('fl_'):
        _mk_fastland()
        _apply5_prev3(name[3:])
    else:
        _apply5_prev3(name)


# ---------------------------------------------------------------------------
# Two-pass pool budgeting.
#
# MIN_CALL_BUDGET fixed the starvation but left the other half on the
# table: at look_ahead=20 a 3.5s primary deadline with a 0.8s floor
# examines 4 of 20 pool items and never looks at the rest. The submission
# probe says that regime is exactly where the leaderboard lives
# (k=1 +0.12, k=10 +7.7, k=20 +13.4; LB 27.27 -> 33).
#
# The fix rests on a property of _first_valid: it walks candidates in
# descending score and returns the first valid one, so a non-None answer
# is already the argmax and a deadline can only ever turn a real answer
# into a false "nothing fits". A cheap pass is therefore lossless -- every
# item it resolves is resolved exactly -- and only the items that ran out
# of time need to be revisited.
#
#   pass 1  survey every (item, container) on a small slice, capped at
#           SURVEY_FRAC of the budget so a late-game state where nothing
#           is cheap cannot eat the whole call
#   pass 2  re-run only the timed-out items, biggest first, on the floor
def _mk_twopass(slice_s, survey_frac, min_call=None):
    import time
    import packer
    vol = lambda s: s['length'] * s['width'] * s['height']

    def choose_action(container_states, pool, time_deadline, relax=False,
                      require_support=True, verifier=None):
        floor = packer.MIN_CALL_BUDGET if min_call is None else min_call
        any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
        items = sorted(pool, key=lambda t: -vol(t[1]))
        pairs = [(pi, spec, ci, cs) for pi, spec in items
                 for ci, cs in container_states.items()]
        start = time.perf_counter()
        survey_end = min(time_deadline, start + (time_deadline - start) * survey_frac)

        best = [None, float('-inf')]

        def offer(pool_idx, cidx, cstate, spec, result):
            pos, orn, score = result
            score += packer.priority_bonus(spec, cstate.geom, any_prio)
            score += packer.W_ITEM_VOL * vol(spec)
            if score > best[1]:
                best[1] = score
                best[0] = (pool_idx, cidx, pos, orn)

        pending = []
        for pool_idx, spec, cidx, cstate in pairs:
            now = time.perf_counter()
            if now > survey_end:
                pending.append((pool_idx, spec, cidx, cstate))
                continue
            per = min(slice_s, survey_end - now)
            r = packer.best_placement(cstate, spec, now + per, relax, require_support)
            if r is not None:
                offer(pool_idx, cidx, cstate, spec, r)
            elif time.perf_counter() - now >= per * 0.9:
                # Consumed its whole slice and reported nothing: that is a
                # deadline, not a verdict. Worth a second, longer look.
                pending.append((pool_idx, spec, cidx, cstate))

        for pool_idx, spec, cidx, cstate in pending:
            now = time.perf_counter()
            if now > time_deadline:
                break
            per = max((time_deadline - now) / max(len(pending), 1), floor)
            r = packer.best_placement(cstate, spec, min(time_deadline, now + per),
                                      relax, require_support)
            if r is not None:
                offer(pool_idx, cidx, cstate, spec, r)
        return best[0]

    packer.choose_action = choose_action


_apply5_prev4 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('two'):
        # two<slice_ms>_<survey_pct>[_<floor_ms>]
        parts = name[3:].split('_')
        sl = float(parts[0]) / 1000.0
        fr = float(parts[1]) / 100.0
        mc = float(parts[2]) / 1000.0 if len(parts) > 2 else None
        _mk_twopass(sl, fr, mc)
    else:
        _apply5_prev4(name)


# ---------------------------------------------------------------------------
# Under-overhang candidates.
#
# _landing returns the highest surface under the footprint, so wherever a
# container's static shelf plate spans the footprint, every candidate is
# forced to rest *on* the shelf. The volume underneath it -- roughly
# 1.9m x 0.65m x 0.75m, about 0.9 m3 or a fifth of the container, and two
# item-layers deep -- is not merely deprioritised, it is unrepresentable.
#
# This adds a second candidate per (x, y, orientation): rest on the
# highest NON-static surface that sits below the lowest static overhang
# spanning the footprint, subject to fitting under that overhang. Purely
# additive to the existing candidate set, so it cannot make the search
# worse by construction, and the oracle still has the final say (the entry
# sweep has to clear the shelf too, which validate_placement checks).
def _mk_undershelf(min_half_x=0.0):
    import packer
    from geometry import GAP
    TOP_TOL = packer.TOP_TOL
    orig_collect = packer._collect

    def under(cstate, cx, cy, hx, hy, inset=0.004):
        """(support_top, supporters, ceiling) below a static overhang.

        min_half_x filters which static plates count as an overhang worth
        going under. Every container -- shelf or not -- carries the narrow
        notch plate at half (0.22, 0.685, 0.02); the space under *that* is
        outside the bottom chamfer near the floor and mostly unplaceable,
        so treating it as an overhang only manufactures low, high-scoring
        candidates that fill the notch strip early. The real shelf is
        half (0.98, 0.323, 0.02), i.e. wide in x, and is the one worth
        reaching under."""
        ex, ey = hx - inset, hy - inset
        ceil = None
        for b in cstate.static_obstacles:
            if b['half'][0] < min_half_x:
                continue
            bc, bh = b['center'], b['half']
            if (abs(cx - bc[0]) < ex + bh[0] and abs(cy - bc[1]) < ey + bh[1]):
                bot = bc[2] - bh[2]
                if ceil is None or bot < ceil:
                    ceil = bot
        if ceil is None:
            return None
        floor = cstate.geom['floor_struct_z']
        if ceil <= floor + 0.05:
            return None                      # not an overhang, just a plate
        support_top = floor
        supporters = None
        limit = floor - TOP_TOL
        for top, bx, bhx, by, bhy, b in cstate.landing_index:
            if top < limit:
                break
            if b['is_static'] or top >= ceil:
                continue
            if abs(cx - bx) >= ex + bhx or abs(cy - by) >= ey + bhy:
                continue
            if supporters is None:
                if top > floor:
                    support_top = top
                limit = support_top - TOP_TOL
                supporters = [b]
            else:
                supporters.append(b)
        return support_top, supporters, ceil

    def collect(cstate, item_spec, relax, xy_fn):
        cands = orig_collect(cstate, item_spec, relax, xy_fn)
        # Bail out before enumerating anything if this container has no
        # overhang worth reaching under. Without this the whole candidate
        # sweep runs and is then thrown away on every non-shelf container,
        # which costs real search depth where the budget is tight: on the
        # large-pool set it measured -2.9 points of items placed on the
        # non-shelf scenes alone.
        if not any(b['half'][0] >= min_half_x for b in cstate.static_obstacles):
            return cands
        g = cstate.geom
        from geometry import ALL_ORNS, half_extents
        length, width, height = item_spec['length'], item_spec['width'], item_spec['height']
        mass = float(item_spec.get('mass', 1.0))
        max_h = g['z_hi'] - g['z_lo']
        walls = packer._infer_walls(cstate)
        extra = []
        for orn in ALL_ORNS:
            hx, hy, hz = half_extents(length, width, height, orn)
            fx, fy, fz = 2 * hx, 2 * hy, 2 * hz
            if fz > max_h:
                continue
            for (x_left, y_back, _wall) in xy_fn(cstate, fx, fy, walls):
                cx, cy = x_left + hx, y_back - hy
                r = under(cstate, cx, cy, hx, hy)
                if r is None:
                    continue
                st, sup, ceil = r
                if sup is not None and not relax:
                    if not all(packer._compatible_support(item_spec, b) for b in sup):
                        continue
                bottom = packer._bottom_for_support(g, st, sup, hz)
                top = bottom + fz
                if top > ceil - GAP or top > g['z_hi'] + 1e-9:
                    continue
                score = (packer.W_BACK * y_back - packer.W_LOW * bottom
                         - packer.W_FLAT * fz - packer.W_LEFT * x_left
                         - packer.W_MASS_HIGH * mass * bottom)
                extra.append((score, cx, cy, bottom + hz, hx, hy, hz, orn, sup, fx, fy))
        if not extra:
            return cands
        out = cands + extra
        out.sort(key=lambda c: -c[0])
        return out

    packer._collect = collect


_apply5_prev5 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('undershelf'):
        _mk_undershelf(float(name[10:]) if len(name) > 10 else 0.0)
    else:
        _apply5_prev5(name)


# ---------------------------------------------------------------------------
# Floor layer is worth zero counted volume.
#
# The official fill_score (evaluator.calculate_fill_rate, fed
# validator.inclusion_margin = -0.005) counts an item only if every corner
# clears every container plane by 5mm. Plane 0 is the floor: outward normal
# (0,0,-1) through z = thickness. An item that settles onto the floor has
# its bottom corners exactly on that plane, so it is rejected. Measured
# over four scenes: 19 of 19 floor-resting items counted zero, while 87 of
# 94 items resting on something else counted (tools/diag_incl.py). That is
# 21% of all placed volume scoring nothing.
#
# So the floor layer's only value is as a platform, and choose_action's
# W_ITEM_VOL bonus -- which deliberately sends the *biggest* item to the
# best-scoring spot -- is spending our largest volumes on the one layer
# that cannot be counted. Dropping the bonus for floor placements lets the
# ordinary positional score decide instead, and that score already
# contains -W_FLAT * height, so the floor naturally attracts the flattest
# items: a thin uncounted platform that lifts everything above it into
# counted territory.
def _mk_floorvol(mult):
    import time
    import packer
    from geometry import half_extents, FLOOR_LIFT
    vol = lambda s: s['length'] * s['width'] * s['height']

    def choose_action(container_states, pool, time_deadline, relax=False,
                      require_support=True, verifier=None):
        any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
        items = sorted(pool, key=lambda t: -vol(t[1]))
        n_calls = max(len(items) * len(container_states), 1)
        best = None
        best_score = float('-inf')
        done = 0
        for pool_idx, spec in items:
            for cidx, cstate in container_states.items():
                now = time.perf_counter()
                left = n_calls - done
                done += 1
                if now > time_deadline:
                    continue
                per = max((time_deadline - now) / max(left, 1), packer.MIN_CALL_BUDGET)
                r = packer.best_placement(cstate, spec, min(time_deadline, now + per),
                                          relax, require_support)
                if r is None:
                    continue
                pos, orn, score = r
                hz = half_extents(spec['length'], spec['width'], spec['height'], orn)[2]
                bottom = pos[2] - hz
                on_floor = bottom <= cstate.geom['floor_struct_z'] + FLOOR_LIFT + 1e-6
                score += packer.priority_bonus(spec, cstate.geom, any_prio)
                score += (mult if on_floor else 1.0) * packer.W_ITEM_VOL * vol(spec)
                if score > best_score:
                    best_score = score
                    best = (pool_idx, cidx, pos, orn)
        return best

    packer.choose_action = choose_action


_apply5_prev6 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('fvol'):
        _mk_floorvol(float(name[4:]) / 100.0)
    else:
        _apply5_prev6(name)


# ---------------------------------------------------------------------------
# Inclusion safety pad. Cannot be driven by ab.py --set: check_inclusion
# binds EFFECTIVE_INCLUSION_MARGIN as a default argument at def time, and
# EFFECTIVE_INCLUSION_MARGIN is itself computed at import time, so setting
# either afterwards is silently ignored.
#
# Targets the second half of the uncounted volume: after the floor plane,
# the rejections are items that drifted out through the back (+Y) and side
# (+X) walls during settling -- 7 items, 6.5% of placed volume, in the
# same four-scene probe. Settling drift is 1-1.6cm and the design pad is
# 1cm, so the two are the same size.
def _mk_incpad(pad):
    import geometry
    margin = geometry.INCLUSION_MARGIN - pad
    orig = geometry.check_inclusion

    def check_inclusion(cgeom, pos_local, half_ext, margin=margin):
        return orig(cgeom, pos_local, half_ext, margin)

    geometry.check_inclusion = check_inclusion
    import packer
    packer.validate_placement.__globals__['check_inclusion'] = check_inclusion


_apply5_prev7 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('pad'):
        _mk_incpad(float(name[3:]) / 1000.0)
    else:
        _apply5_prev7(name)


# ---------------------------------------------------------------------------
# Extra flatness penalty on floor placements only.
#
# Same observation as _mk_floorvol -- the floor layer counts zero -- but
# exploited without touching which item gets chosen (removing the
# bigfirst volume bonus there measured -2.1). If layer one cannot be
# counted, the next best thing is for it to be *thin*, so that more
# counted layers fit above it. This penalises the item's vertical extent
# for floor candidates only, i.e. it picks the flattest orientation of
# whatever item bigfirst already selected.
def _mk_floorflat(w):
    import packer
    orig = packer._collect

    def collect(cstate, item_spec, relax, xy_fn):
        cands = orig(cstate, item_spec, relax, xy_fn)
        out = [((c[0] - w * 2.0 * c[6]) if c[8] is None else c[0],) + tuple(c[1:])
               for c in cands]
        out.sort(key=lambda c: -c[0])
        return out

    packer._collect = collect


_apply5_prev8 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('ffl'):
        _mk_floorflat(float(name[3:]))
    else:
        _apply5_prev8(name)


# ---------------------------------------------------------------------------
# Class-aware pool selection.
#
# scores.placement_scores / soft_item_score penalise a priority or soft item
# whenever ANY item of a different class sits above it and overlaps in x-y --
# not merely one resting on it. So a soft item is safe only if nothing is
# ever stacked over its column, i.e. only if it goes near the top of the
# load. optimize()'s sort keys already push soft items to the end of the
# arrival order, but choose_action -- which is what actually decides the
# order whenever the pool offers a choice -- is completely class-blind: it
# ranks by placement score plus W_ITEM_VOL * volume and nothing else.
#
# Measured cost of that blindness: 8.5 soft violations per scene (of ~36
# items placed), soft_item_score 79.9 and placement_score 83.9 on the dev
# pool. Under the composite objective those two are worth as much as fill.
#
# This defers soft/priority items while any ordinary item is still placeable.
# It is a preference, not a constraint: if only soft items can be placed,
# one still gets placed rather than ending the episode.
def _mk_classdefer(w_soft, w_prio):
    import time
    import packer
    vol = lambda s: s['length'] * s['width'] * s['height']

    def choose_action(container_states, pool, time_deadline, relax=False,
                      require_support=True, verifier=None):
        any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
        items = sorted(pool, key=lambda t: -vol(t[1]))
        n_calls = max(len(items) * len(container_states), 1)
        best = None
        best_score = float('-inf')
        done = 0
        for pool_idx, spec in items:
            for cidx, cstate in container_states.items():
                now = time.perf_counter()
                left = n_calls - done
                done += 1
                if now > time_deadline:
                    continue
                per = max((time_deadline - now) / max(left, 1), packer.MIN_CALL_BUDGET)
                r = packer.best_placement(cstate, spec, min(time_deadline, now + per),
                                          relax, require_support)
                if r is None:
                    continue
                pos, orn, score = r
                score += packer.priority_bonus(spec, cstate.geom, any_prio)
                score += packer.W_ITEM_VOL * vol(spec)
                if spec.get('is_soft'):
                    score -= w_soft
                if spec.get('is_prioritized'):
                    score -= w_prio
                if score > best_score:
                    best_score = score
                    best = (pool_idx, cidx, pos, orn)
        return best

    packer.choose_action = choose_action


_apply5_prev9 = apply5


def apply5(name):  # noqa: F811
    if name.startswith('cls'):
        a = name[3:].split('_')
        _mk_classdefer(float(a[0]), float(a[1]) if len(a) > 1 else float(a[0]))
    else:
        _apply5_prev9(name)
