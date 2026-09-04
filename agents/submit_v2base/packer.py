"""Placement planner built on top of geometry.py's oracle.

Structure of the search (see docs/STRATEGY.md):

* The transport check pushes an item in from the door along -Y then
  slides it in X, clearing obstacles by SAFETY_MARGIN the whole way. So
  the container has to be filled back-to-front, and neighbors in X and Y
  need a real gap -- but *vertical* stacking is free, because the sweep
  travels above the item's final resting height and the final settling
  contact is never collision-checked.

* Only (x, y, orientation) is enumerated; the resting height follows from
  where the item would land. Candidate corners come from extreme points
  (flush with the container, or against an existing box's edge +/- the
  required gap), with a grid sweep as fallback for when the free space
  stops lining up with existing edges.

* Candidates are scored first and validated lazily in score order, so the
  returned placement is exactly the highest-scoring *valid* one while
  usually costing only a handful of oracle calls.

Nothing invalid can ever be returned: geometry.validate_placement is the
final gate on every candidate, so the generator only has to be a good
source of ideas, not perfectly correct.
"""
import time

from geometry import (
    ALL_ORNS, GAP, FLOOR_LIFT, SAFETY_MARGIN,
    half_extents, aabb_half_extents_from_quat, container_geometry,
    validate_placement, resting_eff_start_z,
)

# Box tops within this of each other count as one support level, for
# _landing()'s "what is directly under this candidate" clustering.
TOP_TOL = 0.012
# Separate, looser tolerance for "does this candidate's resulting top
# match a height already used elsewhere in the same wall" (see
# _wall_local_tops / W_LEVEL_MATCH). Deliberately decoupled from TOP_TOL:
# the two once shared a constant, and loosening it to catch more height
# matches also loosened _landing's support clustering, which merges much
# more distant boxes into one "support" and made _footprint_supported's
# bridging check (and therefore the whole search) far more expensive.
# Sized to catch the ~1-3cm spread across the catalog's "flat" resting
# heights (0.20-0.27m across the 7 item types).
WALL_HEIGHT_TOL = 0.035
# Clearance left above a support when resting flush would make the
# transport sweep travel at the item's own height and clip that support.
SUPPORT_LIFT = 0.017
# Hard ceiling on oracle validations per best_placement call. This is a
# runaway guard only -- the real budget is the caller's deadline. It must
# stay well above the typical candidate count: once a slab fills up, every
# candidate in it fails, and the first *valid* placement can sit thousands
# of ranks down the list. Cutting off early there makes best_placement
# report "nothing fits" while good placements remain, which ends the
# episode for no reason.
MAX_VALIDATIONS = 60000

# Scoring weights, tuned by measuring end-to-end fill_score over the two
# sample scenes. Neither pure strategy wins: making height dominate
# (fill each layer completely before going up) and making depth dominate
# (fill the back slab to the ceiling before moving forward) both score
# clearly worse than keeping them within ~2x of each other, which packs
# low and back at the same time and keeps a wide, level surface to build
# the next layer on.
#
# Depth still matters beyond preference: the transport sweep enters from
# the door, so an item placed behind an existing one in the same lane
# would have to path straight through it and gets rejected outright.
#
# Flatness is a tiebreak among orientations -- laying items down keeps
# layer heights closer to uniform and gives each a wider footprint to
# support the layer above.
#
# NB: these were fitted on two scenes only, and the greedy is sensitive
# enough that small changes flip outcomes several points either way. Treat
# them as a decent operating point, not a converged optimum.
W_LOW = 400.0
W_BACK = 200.0
W_FLAT = 100.0
W_LEFT = 5.0
W_MASS_HIGH = 0.6
# Bonus per m^3 of item volume, applied only when the pool offers a
# choice of which item to place next (see choose_action).
W_ITEM_VOL = 3000.0
# Bonus for landing a candidate's top flush with a height already used
# elsewhere *in the same wall* (see _wall_local_tops).
#
# Left at 0.0: a second attempt at the idea reverted in #11.4 (that one
# was scoped globally -- any top anywhere in the container -- and this
# one is scoped to the one wall a candidate's Y actually belongs to, so
# it isn't the same mistake). At weight 250 it measured +21% fill on one
# scene (B, shelf + online-only) but -9% and -29% on two others (D, E),
# for a net *negative* mean across the full 7-scene set (27.8% -> 26.5%
# normalized). Per docs/STRATEGY.md #12.7's own rule ("don't accept off a
# screening subset"), a win on the scene that first showed movement does
# not license shipping it once the full set says otherwise. The
# wall-candidate machinery itself (_infer_walls, _wall_y_candidates) is
# additive to the pre-existing box-edge candidates and measured exactly
# neutral at weight 0 (identical to the pre-wall-building baseline on
# all 7 scenes) -- that part is kept. Only this specific scoring use of
# it is switched off. See docs/STRATEGY.md #13 for the full record.
W_LEVEL_MATCH = 0.0


class ContainerState:
    __slots__ = ('geom', 'boxes', 'floor_boxes', 'static_obstacles')

    def __init__(self, cdict):
        self.geom = container_geometry(cdict)
        self.boxes = []
        for it in cdict.get('packed_items', []):
            if it.get('pos') is None or it.get('orn') is None:
                continue
            cx = it['pos'][0] - self.geom['offset_x']
            hx, hy, hz = aabb_half_extents_from_quat(
                it['length'], it['width'], it['height'], it['orn'])
            self.boxes.append({
                'center': (cx, it['pos'][1], it['pos'][2]), 'half': (hx, hy, hz),
                'is_prioritized': bool(it.get('is_prioritized', False)),
                'is_soft': bool(it.get('is_soft', False)),
                'mass': float(it.get('mass', 1.0)),
                'is_static': False,
            })
        self.static_obstacles = [
            {'center': c, 'half': h, 'is_prioritized': False, 'is_soft': False,
             'mass': 0.0, 'is_static': True}
            for c, h in self.geom['static_boxes']
        ]
        self._refresh()

    def _refresh(self):
        self.floor_boxes = [
            b for b in self.boxes
            if abs((b['center'][2] - b['half'][2]) - self.geom['floor_struct_z']) < 0.03
        ]

    def obstacle_boxes(self):
        return ([(b['center'], b['half']) for b in self.boxes] +
                [(b['center'], b['half']) for b in self.static_obstacles])

    def commit(self, pos_local, half_ext, item_spec):
        self.boxes.append({
            'center': pos_local, 'half': half_ext,
            'is_prioritized': bool(item_spec.get('is_prioritized', False)),
            'is_soft': bool(item_spec.get('is_soft', False)),
            'mass': float(item_spec.get('mass', 1.0)),
            'is_static': False,
        })
        self._refresh()


def _compatible_support(item_spec, box):
    """Placement score penalises a plain item resting on top of a priority
    or soft one, so treat that as a constraint. Same-class stacking
    (priority on priority, soft on soft) carries no penalty."""
    if box['is_static']:
        return True
    if not (box['is_prioritized'] or box['is_soft']):
        return True
    return (bool(item_spec['is_prioritized']) == box['is_prioritized'] and
            bool(item_spec['is_soft']) == box['is_soft'])


def _landing(cstate, cx, cy, hx, hy, inset=0.004):
    """Where an item dropped at (cx, cy) would come to rest: the highest
    top surface under its footprint, and the boxes forming that surface.

    Enumerating discrete "support levels" instead (one per distinct box
    top) fails badly here, because items of mixed heights leave almost
    every box with its own top height -- 19 placed items produced 20
    distinct levels, each backing only one or two boxes, so no candidate
    ever had a surface wide enough to rest on. Asking where an item
    *lands* sidesteps that entirely."""
    ex = hx - inset
    ey = hy - inset
    support_top = cstate.geom['floor_struct_z']
    overlapping = []
    for b in cstate.boxes + cstate.static_obstacles:
        if (abs(cx - b['center'][0]) < ex + b['half'][0] and
                abs(cy - b['center'][1]) < ey + b['half'][1]):
            overlapping.append(b)
            top = b['center'][2] + b['half'][2]
            if top > support_top:
                support_top = top
    if not overlapping:
        return support_top, None
    supporters = [b for b in overlapping
                  if b['center'][2] + b['half'][2] >= support_top - TOP_TOL]
    if not supporters:
        return support_top, None
    return support_top, supporters


def _bottom_for_support(cgeom, support_top, support, half_z):
    """Resting height for an item landing on `support`.

    Flush contact is preferred (no wasted height), but the transport check
    skips its usual 8cm lift whenever the item's underside lands within
    5cm above a *structural* surface (the floor plane or the shelf plane,
    which the simulator tests for regardless of whether a shelf physically
    exists at that x). When that happens the sweep travels at the item's
    own height and clips whatever is actually holding it up, so in those
    cases the item is floated just clear of its support instead and
    allowed to drop the last few millimetres."""
    if support is None:
        # The container floor is never collision-checked during transport,
        # but the floor *is* one of the inclusion planes, so keep clear.
        return cgeom['floor_struct_z'] + FLOOR_LIFT
    if any(b['is_static'] for b in support):
        return support_top + SUPPORT_LIFT
    eff = resting_eff_start_z(cgeom, support_top + half_z, half_z)
    if eff > SAFETY_MARGIN + 0.002:
        return support_top
    return support_top + SUPPORT_LIFT


def _footprint_supported(cx, cy, fx, fy, support, inset=0.004, grid=5,
                         bridge=GAP / 2.0 + 0.003):
    """True iff a grid sample of the footprint is covered by the union of
    `support`, each grown by `bridge`. The bridge allowance spans the
    ordinary inter-item GAP (every neighbouring pair has one by design) --
    a rigid box resting across a 2cm seam between two supports is as
    stable as one resting on a single support. Genuinely open space,
    which is what would tip an item over, is still rejected."""
    hx = max(fx / 2.0 - inset, 0.0)
    hy = max(fy / 2.0 - inset, 0.0)
    for i in range(grid):
        px = cx - hx + (2 * hx) * i / (grid - 1) if grid > 1 else cx
        for j in range(grid):
            py = cy - hy + (2 * hy) * j / (grid - 1) if grid > 1 else cy
            if not any(abs(px - b['center'][0]) <= b['half'][0] + bridge and
                       abs(py - b['center'][1]) <= b['half'][1] + bridge
                       for b in support):
                return False
    return True


def _infer_walls(cstate):
    """Cluster placed items by Y-extent overlap into 'walls' -- see
    docs/STRATEGY.md #12 (George & Robinson 1980 wall-building). A wall
    spans [y_front, y_back] at any X or Z; it is not a persistent object,
    it is re-derived from the actual box layout on every call, so this
    stays correct even after a policy-timeout restart or physics drift
    (nothing here depends on remembered planning state).

    Deliberately excludes the container's static shelf obstacles: the
    small-shelf plate spans nearly the *entire* Y range on its own (it is
    a fixture, not something the algorithm chose the depth of), so
    including it collapses every wall into one covering the whole
    container regardless of height -- confirmed as the cause of a
    complete search failure (zero candidates) when first tried. Shelves
    still constrain candidates exactly as before, through
    geometry.validate_placement; they are just not wall-defining.

    This clustering is order-dependent and not guaranteed optimal -- it
    is only used to *propose* good Y values (see _wall_y_candidates), and
    geometry.validate_placement is still the sole authority on whether a
    candidate is actually legal. An imperfect wall can only cost some
    packing efficiency, never correctness. Returned back-first (largest
    y_back first)."""
    walls = []
    for b in sorted(cstate.boxes, key=lambda b: -(b['center'][1] + b['half'][1])):
        y1 = b['center'][1] + b['half'][1]
        y0 = b['center'][1] - b['half'][1]
        merged = False
        for w in walls:
            if not (y1 < w['y_front'] - 1e-6 or y0 > w['y_back'] + 1e-6):
                w['y_back'] = max(w['y_back'], y1)
                w['y_front'] = min(w['y_front'], y0)
                merged = True
                break
        if not merged:
            walls.append({'y_back': y1, 'y_front': y0})
    walls.sort(key=lambda w: -w['y_back'])
    return walls


def _wall_local_tops(cstate, wall):
    """Top heights already in use *within this specific wall* (its Y-band
    only) -- unlike the reverted W_LEVEL_MATCH (docs/STRATEGY.md #11.4),
    which matched against any top anywhere in the container and measured
    net negative, this is scoped to the one wall a candidate's Y actually
    belongs to. Getting the Y-band to agree (see _wall_y_candidates) is
    necessary but not sufficient for a flat, buildable surface: items
    still pick their own orientation/height independently, so two items
    sharing a wall can still land at different heights unless matching an
    established height is itself worth something in the score. `wall` is
    None for a not-yet-created wall (nothing to match yet)."""
    if wall is None:
        return set()
    tops = set()
    for b in cstate.boxes:
        y0 = b['center'][1] - b['half'][1]
        y1 = b['center'][1] + b['half'][1]
        if y1 < wall['y_front'] - 1e-6 or y0 > wall['y_back'] + 1e-6:
            continue
        tops.add(round(b['center'][2] + b['half'][2], 2))
    return tops


def _wall_y_candidates(cstate, fy, walls=None):
    """Y candidates (item's back edge) aligned to wall structure instead
    of raw box edges: reuse an existing wall (flush with its back edge)
    when the item's depth fits within it, or open a new wall at the
    current frontier (just in front of the shallowest existing wall).

    This -- not the Z/support logic, which is unchanged -- is the fix for
    the fragmentation this module used to produce (19 placed boxes
    forming 20 distinct support-top levels): raw box-edge Y candidates
    wander to a different depth for every box, so _landing() keeps
    finding a slightly different support and nothing lines up into a
    surface wide enough to build on. Constraining Y to wall-back-edges
    makes items that share a wall also share its Y footprint, so
    _landing() naturally finds the same, widening support for later
    items -- coherent layers fall out of the existing landing logic
    instead of needing a separate one.

    Returns (y, wall) pairs -- wall is the originating wall dict (for
    _wall_local_tops), or None for the new-wall/frontier candidate."""
    g = cstate.geom
    if walls is None:
        walls = _infer_walls(cstate)
    out = {}
    for w in walls:
        depth = w['y_back'] - w['y_front']
        if fy <= depth + 1e-6:
            # Clip to the reachable boundary rather than rejecting: an
            # existing item's back edge can sit right at (or, from
            # settling drift, fractionally past) the true wall, beyond
            # the WALL_CLEARANCE-inset g['y_hi'] a *new* candidate must
            # respect. Once clipped it's flush with the container wall,
            # not really "this wall" anymore, so drop the wall link then.
            y = min(w['y_back'], g['y_hi'])
            out[y] = w if y >= w['y_back'] - 1e-6 else None
    frontier = min((w['y_front'] for w in walls), default=g['y_hi'] + 1.0)
    out[min(g['y_hi'], frontier - GAP) if walls else g['y_hi']] = None
    return sorted(((y, w) for y, w in out.items()
                  if y <= g['y_hi'] + 1e-9 and y - fy >= g['y_lo'] - 1e-9),
                  key=lambda t: -t[0])


def _wall_for_y(walls, y, tol=0.01):
    for w in walls:
        if abs(y - w['y_back']) <= tol:
            return w
    return None


def _y_candidates(cstate, fy, walls):
    """Union of classic per-box-edge Y positions (flush with, or GAP
    past, any existing box -- this is what lets several different-depth
    items share one wall's Y-range, each at its own offset, rather than
    only ever sitting flush against the wall's back edge) and the
    wall-derived back-edges from _wall_y_candidates. Restricting Y to
    *only* the wall set was tried and measured worse (docs/STRATEGY.md
    #13): it collapses each wall to a single usable depth and throws away
    all the finer box-edge positions that let a wall's leftover depth
    actually get used. Walls still do real work -- they're what the
    W_LEVEL_MATCH scoring bonus (see _collect) uses to prefer a candidate
    that shares an existing wall's height -- they just no longer gate
    what Y values are considered at all. Returns (y, wall) pairs, back
    (largest y) first."""
    g = cstate.geom
    ys = {g['y_hi'], g['y_lo'] + fy}
    for b in cstate.boxes + cstate.static_obstacles:
        by0 = b['center'][1] - b['half'][1]
        by1 = b['center'][1] + b['half'][1]
        ys.add(by0 - GAP)         # in front of it
        ys.add(by1 + GAP + fy)    # behind it
        ys.add(by1)               # back edges flush
        ys.add(by0 + fy)          # front edges flush
    ys.update(y for y, _w in _wall_y_candidates(cstate, fy, walls))
    ys = sorted((v for v in ys if v <= g['y_hi'] + 1e-9 and v - fy >= g['y_lo'] - 1e-9),
                reverse=True)
    return [(y, _wall_for_y(walls, y)) for y in ys]


def _xy_candidates(cstate, fx, fy, walls):
    """Extreme points for the item's (left, back) corner: Y from
    _y_candidates (wall-aware), X still classic extreme-point (flush
    with, or GAP past, a neighbour's edge) -- there's no structural story
    on X either way, since the transport sweep enters at the item's own
    target X, not a shared corridor (verified against validator.py), so
    X positions never needed to align across items. Yields (x, y, wall)
    triples."""
    g = cstate.geom
    xs = {g['x_lo'], g['x_hi'] - fx}
    for b in cstate.boxes + cstate.static_obstacles:
        bx0 = b['center'][0] - b['half'][0]
        bx1 = b['center'][0] + b['half'][0]
        xs.add(bx1 + GAP)         # to the right of it
        xs.add(bx0 - GAP - fx)    # to the left of it
        xs.add(bx0)               # left edges flush (stacking on top)
        xs.add(bx1 - fx)          # right edges flush
    xs = sorted(v for v in xs if g['x_lo'] - 1e-9 <= v and v + fx <= g['x_hi'] + 1e-9)
    return [(x, y, w) for (y, w) in _y_candidates(cstate, fy, walls) for x in xs]


def _grid_xy(cstate, fx, fy, step, walls):
    """Fallback for when extreme points don't propose anything: sweeps X
    on a fine grid, Y still from _y_candidates (wall-aware)."""
    g = cstate.geom
    x0, x1 = g['x_lo'], g['x_hi'] - fx
    if x0 > x1 + 1e-9:
        return []
    nx = max(int((x1 - x0) / step), 0)
    xs = [x0 + i * step for i in range(nx + 1)] + [x1]
    return [(x, y, w) for (y, w) in _y_candidates(cstate, fy, walls) for x in xs]


def _collect(cstate, item_spec, relax, xy_fn):
    """Build the scored candidate list using `xy_fn` to propose corners.
    The resting height is derived per candidate from where the item would
    land, so only (x, y, orientation) has to be enumerated."""
    g = cstate.geom
    length, width, height = item_spec['length'], item_spec['width'], item_spec['height']
    mass = float(item_spec.get('mass', 1.0))
    max_h = g['z_hi'] - g['z_lo']
    wall_tops_cache = {}
    walls = _infer_walls(cstate)  # computed once, reused for every orientation below

    candidates = []
    for orn in ALL_ORNS:
        hx, hy, hz = half_extents(length, width, height, orn)
        fx, fy, fz = 2 * hx, 2 * hy, 2 * hz
        if fz > max_h:
            continue
        for (x_left, y_back, wall) in xy_fn(cstate, fx, fy, walls):
            cx = x_left + hx
            cy = y_back - hy
            support_top, support = _landing(cstate, cx, cy, hx, hy)
            if support is not None and not relax:
                if not all(_compatible_support(item_spec, b) for b in support):
                    continue
            bottom = _bottom_for_support(g, support_top, support, hz)
            top = bottom + fz
            if top > g['z_hi'] + 1e-9:
                continue
            wall_key = id(wall) if wall is not None else None
            if wall_key not in wall_tops_cache:
                wall_tops_cache[wall_key] = _wall_local_tops(cstate, wall)
            local_tops = wall_tops_cache[wall_key]
            level_bonus = W_LEVEL_MATCH if any(abs(top - t) <= WALL_HEIGHT_TOL for t in local_tops) else 0.0
            score = (W_BACK * y_back - W_LOW * bottom - W_FLAT * fz - W_LEFT * x_left
                     - W_MASS_HIGH * mass * bottom + level_bonus)
            candidates.append((score, cx, cy, bottom + hz, hx, hy, hz, orn, support, fx, fy))
    candidates.sort(key=lambda c: -c[0])
    return candidates


def _first_valid(cstate, candidates, time_deadline):
    g = cstate.geom
    obstacles = cstate.obstacle_boxes()
    checked = 0
    for (score, cx, cy, cz, hx, hy, hz, orn, support, fx, fy) in candidates:
        if checked >= MAX_VALIDATIONS:
            break
        if time_deadline is not None and time.perf_counter() > time_deadline:
            break
        checked += 1
        if support is not None and not _footprint_supported(cx, cy, fx, fy, support):
            continue
        if not validate_placement(g, obstacles, (cx, cy, cz), (hx, hy, hz)):
            continue
        return (cx, cy, cz), orn, score
    return None


def best_placement(cstate, item_spec, time_deadline=None, relax=False):
    """Highest-scoring valid placement for this item, or None.

    Candidates are enumerated and scored first, then validated in score
    order, so the result is the true argmax over valid placements while
    normally costing only a few oracle calls. Extreme points are tried
    first because they pack tightly; a grid sweep is the fallback, since
    on its own the extreme-point set goes empty long before the container
    is actually full.
    """
    result = _first_valid(cstate, _collect(cstate, item_spec, relax, _xy_candidates),
                          time_deadline)
    if result is not None:
        return result
    for step in (0.05, 0.025):
        if time_deadline is not None and time.perf_counter() > time_deadline:
            break
        cands = _collect(cstate, item_spec, relax,
                         lambda cs, fx, fy, walls, _s=step: _grid_xy(cs, fx, fy, _s, walls))
        result = _first_valid(cstate, cands, time_deadline)
        if result is not None:
            return result
    return None


def priority_bonus(item_spec, cgeom, any_priority_container):
    """Route priority baggage into the priority container when one exists."""
    if not any_priority_container:
        return 0.0
    if item_spec['is_prioritized']:
        return 300.0 if cgeom['is_prioritized'] else -600.0
    return -150.0 if cgeom['is_prioritized'] else 0.0


def choose_action(container_states, pool, time_deadline, relax=False):
    """Returns (pool_idx, container_idx, pos_local, orn_idx) or None.

    When the pool offers a choice, big items go first. The placement
    score on its own is purely positional, and a small item can almost
    always reach a lower or further-back spot than a large one, so
    without this the pool is drained smallest-first and the large items
    are left with nowhere to go -- measurably worse than having no
    choice at all. Largest-first is the standard fix.
    """
    any_priority_container = any(cs.geom['is_prioritized'] for cs in container_states.values())
    n_calls = max(len(pool) * len(container_states), 1)
    now = time.perf_counter()
    per_call = max((time_deadline - now) / n_calls, 0.02)

    best = None
    best_score = float('-inf')
    for pool_idx, spec in pool:
        for cidx, cstate in container_states.items():
            now = time.perf_counter()
            if now > time_deadline:
                break
            result = best_placement(cstate, spec, min(time_deadline, now + per_call), relax)
            if result is None:
                continue
            pos, orn, score = result
            score += priority_bonus(spec, cstate.geom, any_priority_container)
            score += W_ITEM_VOL * spec['length'] * spec['width'] * spec['height']
            if score > best_score:
                best_score = score
                best = (pool_idx, cidx, pos, orn)
    return best
