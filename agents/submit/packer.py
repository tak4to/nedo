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
# Static-stability thresholds used by _footprint_supported (the stand-in
# for the evaluator's post-settle displacement check). Swept over the
# 32-scene pool: cover 0.40 -> norm 23.6 but 12/32 episodes end with a
# box toppling; 0.60 -> norm 24.1 with only 5 topples; 0.75 -> norm 21.4,
# back to dying of "nothing fits". 0.60 is the measured optimum.
SUPPORT_MIN_COVER = 0.60
SUPPORT_CENTROID_TOL = 0.35

# Clearance left above a support when resting flush would make the
# transport sweep travel at the item's own height and clip that support.
#
# MUST EXCEED geometry.GAP. That is not a preference, it is the condition
# for the placement to be legal at all: when the 8cm entry lift is
# suppressed the sweep travels at the item's own height, so the only thing
# separating it from the surface holding it up is this lift -- and
# check_transport_path rejects any obstacle closer than GAP.
#
# It was 0.017 against a GAP of 0.022, i.e. 5mm short, which made every
# such placement unconditionally illegal. The cost was invisible because
# it presents as "nothing fits" rather than as an error. The lift is
# suppressed on the floor plane and on the shelf plane
# (geometry.resting_eff_start_z), so this silently deleted:
#   * the entire shelf surface, in every require_shelf container, and
#   * the small notch shelf, which exists in *every* container, and
#   * any item stacked on another whose top happens to land in the
#     shelf plane's 5cm danger band.
# Probed directly at one jammed state: 0 of 187 on-shelf positions passed
# the oracle at 0.017, and 40 of 187 at 0.030 (tools/diag_shelf.py).
#
# Measured over 32 large-pool scenes: +16.2 norm, +19.9 points of items
# placed, better on 32 of 32 and worse on none. The response is flat from
# 0.026 to 0.040 (+14.8 / +16.2 / +16.3 / +15.8), so this sits mid-plateau
# rather than on a tuned peak; below GAP it falls off a cliff, and far
# above it just wastes stack height.
SUPPORT_LIFT = GAP + 0.008

# A static plate counts as an "overhang worth reaching under" only if its
# x half-extent reaches this far. Every container carries the narrow notch
# plate (half 0.22 x 0.685); the space under that sits outside the bottom
# chamfer and is largely unplaceable, so treating it as an overhang only
# manufactures low, high-scoring candidates that clog the notch strip. The
# real shelf is half 0.98 x 0.323 -- wide in x -- and is the one that hides
# usable volume. See _landing_under.
OVERHANG_MIN_HALF_X = 0.5

# Under-overhang candidates: BUILT, MEASURED, AND TURNED OFF.
#
# The volume under a container's shelf really is unrepresentable without
# them (see _landing_under), and on synthetic scenes filling it is worth
# 9-10 points of items placed. It still loses:
#
#   corrected shelf pools   dpacked +9.0 / +10.3  but dCOMPOSITE +0.03 / -0.20
#   corrected mixed pool    dpacked +2.6          dCOMPOSITE +0.10
#   corrected large pool    dpacked +3.5          dCOMPOSITE -0.68
#   REAL sample task 001    packed 64.3% -> 54.8%, composite 58.4 -> 52.6
#                           (deterministic, identical on 3 replicates)
#
# The mechanism: burying items under the shelf buries soft and priority
# ones, so soft_item_score drops ~10 and placement drops with it, cancelling
# the fill and cog gains. Under the leaderboard's step function the change
# looks positive only because it rescues scenes sitting *below* the
# minimum-items threshold -- and both real tasks sit comfortably above it
# (70.7% and 64.3% placed), where it is a straight loss.
#
# Kept behind this flag because the diagnosis is sound and the flag becomes
# right the moment evidence says real tasks fall below the threshold.
USE_UNDER_OVERHANG = False
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
# Penalty per metre of the item's *vertical* extent, i.e. how strongly we
# insist on laying items flat rather than standing them on end.
#
# At the old value of 100 this lost to W_BACK/W_LOW routinely: a suitcase
# stood on end has a smaller footprint, so it reaches a deeper or lower
# landing spot, and 200 x (a few cm deeper) beats the 100 x 0.48m it costs
# to stand up a 0.75m case. Real layouts came out full of 0.76m-tall
# towers, which nothing can be stacked on (fragmentation), raise the
# centre of mass (cog_score) and fall over when shaken (stability_score).
# Measured over the 32-scene pool: 600 is worth +1.1 norm on its own and
# +1.4 on top of the support-criterion fix, 16 scenes better and 8 worse.
W_FLAT = 600.0
# Preference for placing towards the left (small x). Not cosmetic: it is the
# only thing that keeps the door-side notch band reachable.
#
# The transport check enters an item at
#   rel_x = clip(target_x, -L/2 + t + cut_x + hx + start_margin, ...)
# so an item whose target x is left of that bound cannot enter at its own x.
# It enters at the bound and then *slides left*, and that sideways sweep has
# to be clear at its (y, z). Anything already sitting just right of the bound
# at the same height blocks the whole strip behind it -- roughly the leftmost
# 0.45m of the container, about a quarter of the floor.
#
# At the old value of 5 this lost to W_BACK (200) and W_LOW (400) every time,
# and the height maps show the result: a 1.47m stack in the middle with the
# entire notch band still bare floor at 0.04m, unreachable for the rest of the
# episode. Raising it makes the packer take the notch band while the corridor
# to it is still open.
#
# Swept on the 32-scene dev pool and re-checked on 32 held-out scenes.
# Pooled over both (n=64): +2.33 norm, 38 scenes better and 18 worse, sign
# test p = 0.011. The curve is single-peaked -- 900 is clearly too strong
# (-3.3, it starts overriding depth and height) -- but within 150-300 the
# differences are inside the noise, so this is "enough left-preference to
# reach the notch", not a converged optimum.
W_LEFT = 150.0
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
    __slots__ = ('geom', 'boxes', 'floor_boxes', 'static_obstacles', 'landing_index')

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
        # Flattened, top-height-descending view of every obstacle, for
        # _landing. Rebuilt per commit (O(n log n) on <100 boxes) and read
        # tens of thousands of times between commits, so the trade is
        # heavily in favour of precomputing.
        self.landing_index = sorted(
            ((b['center'][2] + b['half'][2], b['center'][0], b['half'][0],
              b['center'][1], b['half'][1], b)
             for b in self.boxes + self.static_obstacles),
            key=lambda e: -e[0])

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
    *lands* sidesteps that entirely.

    This is the hot loop of the whole agent: one call per candidate, and
    profiling one item at a half-full container measured 29086 calls
    taking 46% of best_placement's runtime. It therefore walks
    cstate.landing_index -- boxes pre-flattened out of their dicts and
    sorted by top height, descending -- rather than the box dicts
    themselves. Descending order means the first overlap found is already
    the support surface, so the scan can stop as soon as a box's top
    drops more than TOP_TOL below it: no later box can still be a
    co-supporter. Measured 4.8x faster than the dict version over 90000
    random queries, with identical results on every one.
    """
    ex = hx - inset
    ey = hy - inset
    floor = cstate.geom['floor_struct_z']
    support_top = floor
    supporters = None
    limit = floor - TOP_TOL
    for top, bx, bhx, by, bhy, b in cstate.landing_index:
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


def _landing_under(cstate, cx, cy, hx, hy, inset=0.004):
    """Where an item would rest if it went *under* a static overhang.

    _landing answers "what is the highest surface below this footprint",
    which for any footprint spanning the container's shelf is the shelf
    itself. The volume beneath -- about 1.9 x 0.62 x 0.75 m, a fifth of the
    container and two item-layers deep -- is then not merely deprioritised
    but unrepresentable: no candidate can ever be generated there. Opening
    one jammed shelf scene showed exactly that, four boxes on the shelf and
    the entire space under it empty while the front half was stacked to the
    ceiling.

    Returns (support_top, supporters, ceiling) for the highest NON-static
    surface below the lowest qualifying overhang, or None when there is no
    such overhang over this footprint.

    Measured as an addition to the candidate set (it never removes one):
    on shelf-only scenes it places 8.5 more points of items and lifts 12 of
    32 scenes over the item-count threshold with none falling back; on
    mixed and large-pool scenes it is small but still one-directional.
    """
    ex = hx - inset
    ey = hy - inset
    ceil = None
    for b in cstate.static_obstacles:
        if b['half'][0] < OVERHANG_MIN_HALF_X:
            continue
        bc, bh = b['center'], b['half']
        if abs(cx - bc[0]) < ex + bh[0] and abs(cy - bc[1]) < ey + bh[1]:
            bot = bc[2] - bh[2]
            if ceil is None or bot < ceil:
                ceil = bot
    if ceil is None:
        return None
    floor = cstate.geom['floor_struct_z']
    if ceil <= floor + 0.05:
        return None
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


def _footprint_supported(cx, cy, fx, fy, support, inset=0.004, grid=7,
                         bridge=GAP / 2.0 + 0.003):
    """Stand-in for the evaluator's third gate: `place_item` warps the box
    in, runs SETTLE_STEPS of physics, and fails the episode if the box
    then moved more than `displacement_threshold` or tipped more than
    `angle_displacement_threshold`. geometry.py mirrors the first two
    gates (inclusion, transport) exactly; this is the only one we have to
    approximate, so the approximation's calibration *is* the packing
    density.

    The rule is the textbook static-stability one -- a rigid box stays put
    if its centre of mass projects inside the contact patch:

      1. the item's centre sits over a support,
      2. at least SUPPORT_MIN_COVER of the footprint is carried,
      3. the contact patch's centroid is near the item's centre, so the
         load is not all on one edge.

    The `bridge` allowance spans the ordinary inter-item GAP -- every
    neighbouring pair has one by design, and a rigid box resting across a
    2cm seam between two supports is as stable as one on a single support.

    Measured over the 32-scene pool (docs/DIAGNOSIS.md #1): requiring
    *every* footprint sample to be carried -- what this used to do -- ends
    32/32 episodes with the search reporting "nothing fits" while the real
    validator still accepts placements and 74% of the container is empty.
    Dropping the rule entirely instead ends 25/32 episodes with the box
    toppling. This criterion sits between the two failure modes:
    norm 17.8 -> 24.1, items placed 32.6% -> 39.7%, 26 scenes better and
    3 worse.
    """
    hx = max(fx / 2.0 - inset, 0.0)
    hy = max(fy / 2.0 - inset, 0.0)

    def carried(px, py):
        return any(abs(px - b['center'][0]) <= b['half'][0] + bridge and
                   abs(py - b['center'][1]) <= b['half'][1] + bridge
                   for b in support)

    if not carried(cx, cy):
        return False
    hit = 0
    total = 0
    sx = sy = 0.0
    for i in range(grid):
        px = cx - hx + (2 * hx) * i / (grid - 1) if grid > 1 else cx
        for j in range(grid):
            py = cy - hy + (2 * hy) * j / (grid - 1) if grid > 1 else cy
            total += 1
            if carried(px, py):
                hit += 1
                sx += px
                sy += py
    if hit < SUPPORT_MIN_COVER * total:
        return False
    # Load centroid must be near the item's own centre, else it tips.
    return (abs(sx / hit - cx) <= hx * SUPPORT_CENTROID_TOL and
            abs(sy / hit - cy) <= hy * SUPPORT_CENTROID_TOL)


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
    # Checked once, before any enumeration: without this the under-overhang
    # sweep runs in full and is discarded on every container that has no
    # shelf, which costs real search depth where the budget is tight (-2.9
    # points of items placed on the large-pool set's non-shelf scenes).
    has_overhang = USE_UNDER_OVERHANG and any(b['half'][0] >= OVERHANG_MIN_HALF_X
                                              for b in cstate.static_obstacles)

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
    if has_overhang:
        for orn in ALL_ORNS:
            hx, hy, hz = half_extents(length, width, height, orn)
            fx, fy, fz = 2 * hx, 2 * hy, 2 * hz
            if fz > max_h:
                continue
            for (x_left, y_back, _wall) in xy_fn(cstate, fx, fy, walls):
                cx = x_left + hx
                cy = y_back - hy
                found = _landing_under(cstate, cx, cy, hx, hy)
                if found is None:
                    continue
                support_top, support, ceil = found
                if support is not None and not relax:
                    if not all(_compatible_support(item_spec, b) for b in support):
                        continue
                bottom = _bottom_for_support(g, support_top, support, hz)
                top = bottom + fz
                if top > ceil - GAP or top > g['z_hi'] + 1e-9:
                    continue
                score = (W_BACK * y_back - W_LOW * bottom - W_FLAT * fz - W_LEFT * x_left
                         - W_MASS_HIGH * mass * bottom)
                candidates.append((score, cx, cy, bottom + hz, hx, hy, hz, orn, support, fx, fy))
    candidates.sort(key=lambda c: -c[0])
    return candidates


def _first_valid(cstate, candidates, time_deadline, require_support=True, verify=None):
    g = cstate.geom
    obstacles = cstate.obstacle_boxes()
    checked = 0
    demoted = None
    for (score, cx, cy, cz, hx, hy, hz, orn, support, fx, fy) in candidates:
        if checked >= MAX_VALIDATIONS:
            break
        if time_deadline is not None and time.perf_counter() > time_deadline:
            break
        checked += 1
        if (require_support and support is not None
                and not _footprint_supported(cx, cy, fx, fy, support)):
            continue
        if not validate_placement(g, obstacles, (cx, cy, cz), (hx, hy, hz)):
            continue
        if verify is not None and not verify((cx, cy, cz), orn):
            # The physics twin says this one collapses. Keep walking the
            # list, but remember it: the twin is a *preference*, never a
            # veto. Dropping the candidate outright would sometimes leave
            # the caller with nothing at all, and "nothing" ends the
            # episode, which is strictly worse than a placement that might
            # topple. Demoting instead makes the gate structurally unable
            # to lose -- the worst case is the answer we would have given
            # without it.
            if demoted is None:
                demoted = ((cx, cy, cz), orn, score)
            continue
        return (cx, cy, cz), orn, score
    return demoted


def best_placement(cstate, item_spec, time_deadline=None, relax=False, require_support=True,
                   verify=None):
    """Highest-scoring valid placement for this item, or None.

    Candidates are enumerated and scored first, then validated in score
    order, so the result is the true argmax over valid placements while
    normally costing only a few oracle calls. Extreme points are tried
    first because they pack tightly; a grid sweep is the fallback, since
    on its own the extreme-point set goes empty long before the container
    is actually full.
    """
    result = _first_valid(cstate, _collect(cstate, item_spec, relax, _xy_candidates),
                          time_deadline, require_support, verify)
    if result is not None:
        return result
    for step in (0.05, 0.025):
        if time_deadline is not None and time.perf_counter() > time_deadline:
            break
        cands = _collect(cstate, item_spec, relax,
                         lambda cs, fx, fy, walls, _s=step: _grid_xy(cs, fx, fy, _s, walls))
        result = _first_valid(cstate, cands, time_deadline, require_support, verify)
        if result is not None:
            return result
    return None


def placement_at(cstate, item_spec, cx, cy, orn, relax=False):
    """Re-derive a *planned* (x, y, orientation) against the state we can
    actually see, and validate it.

    Not used by the shipped policy -- replaying optimize()'s plan measured
    worse than simply re-running the greedy (see agent.Agent.__init__ for
    the numbers and the reason). It is kept because a placement-level
    offline search (docs/GAMEPLAN.md #3.3c) needs exactly this primitive:
    a way to score a *chosen* position against the observed state rather
    than only ever taking the greedy argmax.

    optimize() plans on design coordinates, but every placed item settles
    by 1-2cm before the next call, so a plan replayed verbatim drifts out
    of spec (measured in docs/STRATEGY.md #2.3). Only (x, y, orn) is
    replayed; the resting height is recomputed from the observed boxes,
    exactly as best_placement does for a fresh candidate. That way the
    plan survives drift instead of being invalidated by it.

    Returns (pos_local, orn, score) or None if the plan no longer holds --
    in which case the caller falls back to the ordinary greedy search.
    """
    g = cstate.geom
    hx, hy, hz = half_extents(item_spec['length'], item_spec['width'],
                              item_spec['height'], orn)
    fx, fy, fz = 2 * hx, 2 * hy, 2 * hz
    if fz > g['z_hi'] - g['z_lo']:
        return None
    support_top, support = _landing(cstate, cx, cy, hx, hy)
    if support is not None and not relax:
        if not all(_compatible_support(item_spec, b) for b in support):
            return None
    bottom = _bottom_for_support(g, support_top, support, hz)
    if bottom + fz > g['z_hi'] + 1e-9:
        return None
    if support is not None and not _footprint_supported(cx, cy, fx, fy, support):
        return None
    pos = (cx, cy, bottom + hz)
    if not validate_placement(g, cstate.obstacle_boxes(), pos, (hx, hy, hz)):
        return None
    score = (W_BACK * (cy + hy) - W_LOW * bottom - W_FLAT * fz - W_LEFT * (cx - hx)
             - W_MASS_HIGH * float(item_spec.get('mass', 1.0)) * bottom)
    return pos, orn, score


def priority_bonus(item_spec, cgeom, any_priority_container):
    """Route priority baggage into the priority container when one exists."""
    if not any_priority_container:
        return 0.0
    if item_spec['is_prioritized']:
        return 300.0 if cgeom['is_prioritized'] else -600.0
    return -150.0 if cgeom['is_prioritized'] else 0.0


# Floor on the time handed to a single best_placement call, in seconds.
#
# The budget used to be split evenly over every (pool item x container)
# pair. That is fine at look_ahead=1 and starves the search at
# look_ahead=20: 3.5s / 18 items = 0.19s per call, against a measured
# 0.25-0.35s for one best_placement at a half-full container
# (tools/diag_prof.py). Every call then times out, choose_action returns
# None, and the agent dies reporting "nothing fits" -- at a state where a
# 3cm brute-force sweep finds 13543 valid placements (tools/diag_gap.py
# on P00, which died with 22 of 40 items placed and 2.0 m3 of headroom
# still above the skyline).
#
# It is a compute-allocation bug, not a packing one, and it was worth as
# much as every scoring change put together: handing the unmodified agent
# a 15s budget scores +2.78 norm on the 32-scene pool, and this floor
# recovers +2.37 of that inside the original 5s (holdout: +2.27, dpct
# +2.4, 10 scenes better and 5 worse).
#
# Evaluating fewer pool items properly beats evaluating all of them
# badly, so the tail of a large pool simply goes unexamined. It is
# examined biggest-first, which is the order the W_ITEM_VOL term prefers
# anyway. Measured over 0.4 / 0.8 / 1.5 / 2.5s the response is flat from
# 0.8s on (+1.76 / +2.37 / +1.71 / +2.21), so this sits at the knee.
MIN_CALL_BUDGET = 0.8


def choose_action(container_states, pool, time_deadline, relax=False, require_support=True,
                  verifier=None):
    """Returns (pool_idx, container_idx, pos_local, orn_idx) or None.

    When the pool offers a choice, big items go first. The placement
    score on its own is purely positional, and a small item can almost
    always reach a lower or further-back spot than a large one, so
    without this the pool is drained smallest-first and the large items
    are left with nowhere to go -- measurably worse than having no
    choice at all. Largest-first is the standard fix, and it doubles as
    the order in which the pool gets truncated when MIN_CALL_BUDGET
    means there is not enough time to look at all of it.
    """
    any_priority_container = any(cs.geom['is_prioritized'] for cs in container_states.values())
    items = sorted(pool, key=lambda t: -(t[1]['length'] * t[1]['width'] * t[1]['height']))
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
            per_call = max((time_deadline - now) / max(left, 1), MIN_CALL_BUDGET)
            verify = None if verifier is None else verifier(cidx, spec)
            result = best_placement(cstate, spec, min(time_deadline, now + per_call), relax,
                                    require_support, verify)
            if result is None:
                continue
            pos, orn, score = result
            score += priority_bonus(spec, cstate.geom, any_priority_container)
            score += W_ITEM_VOL * spec['length'] * spec['width'] * spec['height']
            if score > best_score:
                best_score = score
                best = (pool_idx, cidx, pos, orn)
    return best
