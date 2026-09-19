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
SUPPORT_MIN_COVER = 0.62978
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
W_LOW = 415.8
W_BACK = 326.87
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
W_FLAT = 1368.9
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
W_LEFT = 130.16
W_MASS_HIGH = 0.50605
# Bonus per m^3 of item volume, applied only when the pool offers a
# choice of which item to place next (see choose_action).
W_ITEM_VOL = 1351.2
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

# ---------------------------------------------------------------------------
# Extra placement features, all shipping at weight 0.
#
# Every one of these encodes an idea that was tried on its own and lost --
# heightmap minimisation, support ratio, wall contact, interlocking,
# ceiling packing, deferring soft/priority items. Twenty-three such
# single-term experiments went 0-for-23. That record is not evidence the
# ideas are worthless; it is evidence that adding one term while holding
# six hand-tuned weights fixed is the wrong experiment, because the
# existing weights were fitted without it. Carrying them as *features*
# instead lets tools/cem.py fit the whole vector jointly against the real
# objective (composite through the item-count cliff), which no amount of
# one-at-a-time testing can reach.
#
# At 0.0 the code below is a strict no-op: the guards skip the computation
# entirely, so shipping this changes nothing until a weight is set.
SEAM_TOL = 0.03          # edge alignment closer than this counts as a seam
WALL_TOL = 0.03          # flush against a container bound within this
SIDE_TOL = 0.03          # face-to-face gap closer than this counts as side contact
W_WASTE = 107.04            # penalty per metre of mean void depth under the footprint
W_SUPPORT = 222.17          # bonus per unit of footprint actually carried (0..1)
W_WALL = 67.088             # bonus per container wall the item lands flush against
W_SEAM = -332.72             # penalty when the item's edges line up with its support's
W_CEIL = 469.29             # weight on the clearance left above the item
# Bonus per unit of side surface area touching a *neighbouring item* (not a
# container wall -- that is W_WALL, and above 0.0 ships as a strict no-op
# like the rest of this block).
#
# Added 2026-09-13 from a mechanism study, not a hunch: docs/2026-09-13-
# survey戦略.md P2 instrumented 801 placed items (tools/mechanism_log.py)
# with real settle-and-shake displacement and found this is the single
# clearest lever measured so far -- larger than height in the stack, larger
# than container-wall contact (which was flat across 0/1/2 walls: 0.132-
# 0.142m disp), larger than floor contact (which only separates floor from
# everything else, 0.009m vs 0.164m, and every item not on the floor still
# needs *something* to hold it):
#
#   isolated (no side-adjacent item, <1cm^2 contact)   0.165m disp, 78% moved
#   some side contact with a neighbour                 0.104m disp, 56% moved
#
# i.e. what keeps a non-floor item still under shake is not the container
# wall, it's another item pressed against its side.
W_SIDE = 0.0
W_SOFT_DEFER = -269.02       # pool-selection penalty for soft items
W_PRIO_DEFER = 481.4       # pool-selection penalty for priority items
WASTE_CELL = 0.03        # heightmap resolution for the waste feature


def _waste_depths(cstate, cands):
    """Mean void depth under each candidate's footprint, vectorised.

    Wang & Hauser's heightmap-minimisation term. W_LOW already penalises a
    high resting height but cannot tell apart an item sitting high on a
    solid stack from one whose corner clips a neighbour so most of its
    underside bridges air -- and the second is what traps volume. Measured
    at the current operating point, 0.39-1.31 m3 per container sits in
    voids below the skyline, which is why this is worth carrying as a
    feature even though an earlier fixed-weight version of it lost (it was
    measured when the free space was above the skyline, not below).

    A summed-area table over a coarse heightmap makes each lookup O(1).
    """
    import numpy as np
    g = cstate.geom
    x0, x1 = g['x_lo'], g['x_hi']
    y0, y1 = g['y_lo'], g['y_hi']
    nx = max(int((x1 - x0) / WASTE_CELL) + 1, 1)
    ny = max(int((y1 - y0) / WASTE_CELL) + 1, 1)
    H = np.full((nx, ny), g['floor_struct_z'], dtype=np.float64)
    for b in cstate.boxes + cstate.static_obstacles:
        bcx, bcy, bcz = b['center']
        bhx, bhy, bhz = b['half']
        i0 = max(int((bcx - bhx - x0) / WASTE_CELL), 0)
        i1 = min(int((bcx + bhx - x0) / WASTE_CELL) + 1, nx)
        j0 = max(int((bcy - bhy - y0) / WASTE_CELL), 0)
        j1 = min(int((bcy + bhy - y0) / WASTE_CELL) + 1, ny)
        if i1 > i0 and j1 > j0:
            np.maximum(H[i0:i1, j0:j1], bcz + bhz, out=H[i0:i1, j0:j1])
    S = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    S[1:, 1:] = H.cumsum(0).cumsum(1)
    cx = np.fromiter((c[1] for c in cands), float, len(cands))
    cy = np.fromiter((c[2] for c in cands), float, len(cands))
    hx = np.fromiter((c[4] for c in cands), float, len(cands))
    hy = np.fromiter((c[5] for c in cands), float, len(cands))
    hz = np.fromiter((c[6] for c in cands), float, len(cands))
    cz = np.fromiter((c[3] for c in cands), float, len(cands))
    i0 = np.clip(((cx - hx - x0) / WASTE_CELL).astype(np.int64), 0, nx - 1)
    i1 = np.maximum(np.clip(((cx + hx - x0) / WASTE_CELL).astype(np.int64) + 1, 1, nx), i0 + 1)
    j0 = np.clip(((cy - hy - y0) / WASTE_CELL).astype(np.int64), 0, ny - 1)
    j1 = np.maximum(np.clip(((cy + hy - y0) / WASTE_CELL).astype(np.int64) + 1, 1, ny), j0 + 1)
    tot = S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]
    mean_h = tot / ((i1 - i0) * (j1 - j0))
    return np.maximum((cz - hz) - mean_h, 0.0)


def _side_contact_frac(cstate, cx, cy, hx, hy, bottom, top, fx, fy, fz):
    """Fraction (0..1) of this candidate's side surface touching a
    *placed item* (not a container wall or shelf -- W_WALL already covers
    walls, and docs/2026-09-13-survey戦略.md P2 found wall contact flat
    across 0-2 walls while item-to-item contact was the strongest single
    lever measured).

    O(number of placed items); only called when W_SIDE != 0, so it costs
    nothing at the shipped weight. cstate.static_obstacles is deliberately
    excluded -- this is the same scope as the mechanism study that
    motivated it.
    """
    area = 0.0
    for b in cstate.boxes:
        bc, bh = b['center'], b['half']
        z_overlap = min(top, bc[2] + bh[2]) - max(bottom, bc[2] - bh[2])
        if z_overlap <= 0.0:
            continue
        zc = min(z_overlap, fz)
        if (abs((cx - hx) - (bc[0] + bh[0])) < SIDE_TOL
                or abs((cx + hx) - (bc[0] - bh[0])) < SIDE_TOL):
            oy = min(cy + hy, bc[1] + bh[1]) - max(cy - hy, bc[1] - bh[1])
            if oy > 0.0:
                area += oy * zc
        if (abs((cy - hy) - (bc[1] + bh[1])) < SIDE_TOL
                or abs((cy + hy) - (bc[1] - bh[1])) < SIDE_TOL):
            ox = min(cx + hx, bc[0] + bh[0]) - max(cx - hx, bc[0] - bh[0])
            if ox > 0.0:
                area += ox * zc
    total_side = 2.0 * (fx + fy) * fz
    return min(area / total_side, 1.0) if total_side > 0.0 else 0.0


def _shape_features(g, support, cx, cy, hx, hy, fx, fy, top):
    """(support_frac, wall_contact, seam_align, ceil_gap) for one candidate.

    All O(number of supporting boxes), which is small; nothing here walks
    the full box list.
    """
    frac = 0.0
    seam = 0.0
    if support:
        for b in support:
            bc, bh = b['center'], b['half']
            ox = min(cx + hx, bc[0] + bh[0]) - max(cx - hx, bc[0] - bh[0])
            oy = min(cy + hy, bc[1] + bh[1]) - max(cy - hy, bc[1] - bh[1])
            if ox > 0.0 and oy > 0.0:
                frac += ox * oy
            if (abs((cx - hx) - (bc[0] - bh[0])) < SEAM_TOL
                    or abs((cx + hx) - (bc[0] + bh[0])) < SEAM_TOL
                    or abs((cy - hy) - (bc[1] - bh[1])) < SEAM_TOL
                    or abs((cy + hy) - (bc[1] + bh[1])) < SEAM_TOL):
                seam = 1.0
        frac = min(frac / (fx * fy), 1.0)
    walls = 0
    if cx - hx <= g['x_lo'] + WALL_TOL:
        walls += 1
    if cx + hx >= g['x_hi'] - WALL_TOL:
        walls += 1
    if cy + hy >= g['y_hi'] - WALL_TOL:
        walls += 1
    if cy - hy <= g['y_lo'] + WALL_TOL:
        walls += 1
    return frac, float(walls), seam, g['z_hi'] - top


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

    def clone(self):
        """A copy that can be committed to without disturbing this one.

        `boxes` is the only field commit() mutates, and the derived views
        (`floor_boxes`, `landing_index`) are rebuilt wholesale by _refresh
        rather than edited in place -- so a clone can share them until it
        commits something of its own. That makes a snapshot O(len(boxes))
        with no re-sort, which is what makes it affordable for optimize()
        to keep one per position along a rollout.
        """
        c = ContainerState.__new__(ContainerState)
        c.geom = self.geom
        c.boxes = list(self.boxes)
        c.static_obstacles = self.static_obstacles
        c.floor_boxes = self.floor_boxes
        c.landing_index = self.landing_index
        return c

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


# Record settled heights, not design heights, inside optimize()'s rollout.
#
# _bottom_for_support deliberately floats some items SUPPORT_LIFT (0.030)
# clear of whatever holds them up -- and FLOOR_LIFT (0.020) above the floor
# -- because the transport sweep would otherwise clip the support. The
# evaluator then drops the item and it settles flush. Measured over 637
# accepted placements (tools/drift_fit.py): items move -0.0118 m on average
# in z and essentially nothing in x or y, and the distribution is not noise
# but two spikes -- 70% of rigid items do not move at all (they were placed
# flush) and the rest sink by almost exactly 0.020 or 0.030.
#
# The rollout used to commit the design height and never move it, so every
# later item in that column was stacked on a surface 2-3cm higher than the
# one the real run offers, compounding up the stack. policy() must still
# *ask* for the design height -- that is what makes the placement legal --
# so this correction is only for the rollout's own bookkeeping.
ROLLOUT_SETTLE = 1.0

# Salt for optimize()'s ruin-and-recreate RNG. Ships at 0; the search is
# seeded from the item indices so a scene always searches the same sequence.
# Exists so that one scene's *sensitivity* can be measured -- running the
# same configuration under several salts says how much of a single-scene
# result is the change and how much is which local optimum the walk fell
# into. Without it a real task, of which there is exactly one in offline
# mode, cannot be told apart from a coin flip.
OFFLINE_SEED_SALT = 0.0

# How many independent ruin-and-recreate walks optimize() runs, sharing one
# global record.
#
# The walk's outcome is a lottery. Running the same configuration on the one
# real offline task under eight different search seeds spreads composite
# over 54.9 to 64.9 (sd 2.96) and items placed over 11.1 percentage points
# -- so which local optimum the walk happens to fall into matters more than
# most of the changes measured against it, and a single walk banks whichever
# one it drew. That is also what the search has spare budget for: with a
# single walk the record typically stops improving in the first tens of
# seconds and the remaining ~140s of the 150s budget changes nothing.
#
# Restarting converts the lottery into a maximum -- of the *objective*.
#
# MEASURED AND TURNED OFF. Eight walks against one, over eight search seeds
# on R000: composite -0.32 (se 2.48, 4W/4L), and the spread across seeds
# went up rather than down (sd 3.68 -> 4.70).
#
# This is the fourth thing that spends more of the budget and does not pay
# (record-to-record travel, the prefix cache, and simply searching longer
# were the others), and they all fail for one reason. The rollout ranks two
# arrival orders for the same scene at Spearman +0.49 on composite
# (tools/order_fidelity.py). Taking the best of N candidates under a ranking
# that noisy selects the order whose rollout score is most optimistically
# wrong about as fast as it selects a genuinely better one -- the winner's
# curse grows with N alongside the true maximum, so the net is nil and the
# variance rises.
#
# The corollary is worth stating, because it decides what to try next: no
# amount of extra search buys anything at this fidelity. The two changes
# that did pay both improved the *ranking* rather than the number of
# candidates ranked -- OFFLINE_TIEBREAK changed what the objective measures,
# ROLLOUT_SETTLE made the simulated state match the real one. Beam search,
# ALNS operator tuning and BRKGA are all "more candidates" and should wait
# until the rollout ranks at 0.7-0.8.
OFFLINE_RESTARTS = 1.0


def settled_center_z(cstate, cx, cy, half):
    """Centre height an item placed at (cx, cy) will settle to: flush on
    whatever _landing says is underneath it."""
    top, _ = _landing(cstate, cx, cy, half[0], half[1])
    return top + half[2]


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
    # Both guards are checked once, up front: at the shipped weights of 0
    # neither block runs at all, so the extra features cost nothing until
    # something actually sets them.
    use_shape = bool(W_SUPPORT or W_WALL or W_SEAM or W_CEIL)
    use_side = bool(W_SIDE)  # separate gate: O(n_boxes), not O(n_support)

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
            if use_shape:
                # NB: not `walls` -- that name holds _infer_walls' result,
                # which xy_fn needs on every later iteration.
                sfrac, wallc, seam, cgap = _shape_features(
                    g, support, cx, cy, hx, hy, fx, fy, top)
                score += (W_SUPPORT * sfrac + W_WALL * wallc
                          - W_SEAM * seam - W_CEIL * cgap)
            if use_side:
                score += W_SIDE * _side_contact_frac(
                    cstate, cx, cy, hx, hy, bottom, top, fx, fy, fz)
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
    if W_WASTE and candidates:
        depths = _waste_depths(cstate, candidates)
        candidates = [(c[0] - W_WASTE * float(d),) + c[1:]
                      for c, d in zip(candidates, depths)]
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


# --- Offline search objective (used by agent.optimize's rollout) ---------
#
# What a candidate arrival order is worth. Until 2026-09-08 this was
# (items placed, volume placed): the search maximised *count and fill*.
# Three leaderboard observations say that is the wrong quantity --
#
#   B -> C moved fill +16.6 and items placed +26.5pt and the leaderboard
#   +1.72. Everything the old objective measures is saturated; the only
#   metrics with room left are stability and cog.
#
# -- and the volume tiebreak is not merely inert, it points the wrong way:
# ranking ties by "most volume placed" prefers putting the big boxes down
# first, which stacks higher, which lowers cog.
#
# Measured on R000 at the full 150s budget (tools/probe_opt.py): the search
# visits 214 orders that all place 30 items, their cog spans 48.45 to
# 53.05, and the volume tiebreak hands back the 49.08 one. Ranking those
# same 214 by the composite instead is worth +3.97 cog for free.
#
# The count stays the *primary* key (agent.py builds the tuple), so this
# only ever decides ties -- an order can never be preferred for placing
# fewer items, which is what keeps the scoring cliff out of reach.
#
# Measured over 64 offline scenes (optimize=True, look_ahead=1) against the
# volume tiebreak: composite +1.68 (se 0.63, 38W/25L) with items placed
# unchanged (+0.01pt), and +1.58 (se 0.60, 38W/26L) on a seed-separated
# holdout -- so the gain transfers rather than being a fit to the pool. The
# gain is not where it was predicted: cog moved +0.03, and it came from
# placement (+5.91) and soft (+3.55). Arrival order decides what ends up
# stacked on a priority or soft item, and the volume tiebreak was blind to
# it.
OFFLINE_TIEBREAK = 1.0      # 1 = composite proxy, 0 = the pre-2026-09-08 volume

# Record-to-record travel, the acceptance rule for the ruin-and-recreate
# phase. Santini, Ropke & Hvattum (J. Heuristics 2018) rank SA, threshold
# acceptance and RRT as the top group for ALNS, with RRT undominated;
# accepting only strict improvements -- what optimize() did until now --
# is not in the comparison at all. It showed: on R000, 820 rollouts
# produced 4 accepted moves, the last at t=9.7s of 150.
#
# A candidate becomes the search's current position when
#   items * RRT_ITEM_WORTH + composite  >=  the same for the record  - RRT_DEV.
#
# MEASURED AND TURNED OFF. At RRT_DEV = 1.0 -- just under the composite
# spread observed within one count level -- it costs 0.68 composite against
# the same configuration with RRT_DEV = 0 (64 offline scenes, se 0.55,
# 32W/31L, and 0.49pt fewer items placed). The freeze it was built to fix is
# real, but crossing the plateau is not what the search was short of: once
# the objective could tell plateau solutions apart (OFFLINE_TIEBREAK), a
# strict-improvement walk found them on its own, and letting the walk drift
# downhill only cost placements.
#
# Left switchable rather than deleted, because the acceptance rule is worth
# revisiting if the objective changes again.
RRT_DEV = 0.0
RRT_ITEM_WORTH = 10.0

# Reuse the simulated prefix a trial shares with the order it came from.
# Ruin-and-recreate moves items around inside an order, so consecutive
# trials agree on a long head -- measured on R000, 64% of the items and 60%
# of a rollout's wall time -- and re-simulating it buys nothing. This is
# exactly semantics-preserving (tools/check_resume.py re-runs every resumed
# rollout from scratch and compares key, order and plan); all it buys is
# rollouts.
#
# MEASURED AND TURNED OFF. The prefix really is 64% of the items and 60% of
# a rollout's time, and the extra rollouts are real, but they do not buy a
# better answer:
#
#   64 offline scenes, old objective   composite -0.64 (se 0.52, 12W/20L)
#   64 offline scenes, new objective   composite +0.42 (se 0.39, 21W/18L)
#   real task R000, old objective      58.9 -> 58.9  (identical order)
#   real task R000, new objective      62.2 -> 55.8
#
# Neither pool result clears its own standard error, and re-measured over
# eight search seeds on R000 it is +0.14 (se 0.51, 3W/2L, three seeds
# byte-identical) -- neutral. It was briefly turned off as *harmful* on the
# strength of one R000 run showing -6.4; that was one draw from a scene
# whose own spread across seeds is sd 2.96 composite, so it said nothing.
# See OFFLINE_SEED_SALT.
#
# It stays off because nothing argues for it, not because it hurts: a pure
# budget optimisation has no mechanism by which it improves the *answer*,
# it only moves the search elsewhere in the plateau, and off is the simpler
# behaviour. Worth revisiting whenever the rollout gets closer still to the
# run it predicts -- extra rollouts are worth more the better it ranks.
OFFLINE_PREFIX_CACHE = 0.0


def usable_volume(g):
    """Container.volume, replicated from src/ground_handling/containers.py.

    The observation dict does carry 'volume', but container_geometry does
    not keep it and the policy's observation is not guaranteed to, so it is
    rederived from the same seven numbers the evaluator uses.
    """
    il = g['length'] - 2.0 * g['thickness']
    iw = g['width'] - 2.0 * g['thickness']
    ih = g['height'] - g['thickness'] - g['buffer']
    vol = il * iw * ih
    vol -= 0.5 * (g['cut_x'] - g['thickness']) * (g['cut_y'] - g['thickness']) * iw
    vol -= g['cut_x'] * g['thickness'] * iw
    if g['has_shelf']:
        vol -= il * g['thickness'] * (g['width'] / 2.0 - 2.0 * g['thickness'])
    return max(vol, 1e-9)


def _class_score(boxes, attr, check_container):
    """placement_score / soft_item_score, mirroring tools/scores.py.

    A member of the class is penalised once when an item of the *other*
    class rests on top of it (same-class stacking is free), and once more,
    for priority items only, when it sits in a container that is not the
    designated priority one.
    """
    members = [(cs, b) for cs, b in boxes if b[attr]]
    if not members:
        return 100.0
    bad = 0
    for cs, b in members:
        top = b['center'][2] + b['half'][2]
        for ocs, o in boxes:
            if o is b or ocs is not cs or o[attr] == b[attr]:
                continue
            if o['center'][2] - o['half'][2] < top - 0.02:
                continue
            if (abs(o['center'][0] - b['center'][0]) < o['half'][0] + b['half'][0] and
                    abs(o['center'][1] - b['center'][1]) < o['half'][1] + b['half'][1]):
                bad += 1
                break
    wrong = 0
    if check_container:
        wrong = sum(1 for cs, b in members if not cs.geom['is_prioritized'])
    return 100.0 * (1.0 - (bad + wrong) / (2.0 * len(members)))


def pack_metrics(states):
    """The four scored metrics a rollout can work out for itself.

    fill, cog, placement and soft are deterministic functions of which box
    ended up where, and a rollout's ContainerStates already carry every
    input (centre, half extents, mass, is_soft, is_prioritized). Only
    stability needs physics, so it is not in here; the return value is the
    mean of the four, which is a monotone stand-in for the composite for
    the purpose of ranking two orders.

    One deliberate departure from the evaluator, on fill: official
    fill_score counts an item only when all eight corners clear every plane
    by 5mm (evaluator.calculate_fill_rate runs at inclusion_margin -0.005),
    and a box resting on the floor never can -- its underside sits at
    exactly `thickness`. Measured: 0 of 19 floor-contact items counted
    against 87 of 94 stacked ones (docs/WORKLOG.md Day 4c). So the floor
    layer is excluded rather than credited with volume it will not score.
    """
    denom = 0.0
    counted_vol = 0.0
    mass = 0.0
    moment = 0.0
    height = 0.0
    any_prio_c = False
    boxes = []
    for cs in states.values():
        g = cs.geom
        denom += usable_volume(g)
        if g['height'] > height:
            height = g['height']
        any_prio_c = any_prio_c or g['is_prioritized']
        floor = {id(b) for b in cs.floor_boxes}
        for b in cs.boxes:
            hx, hy, hz = b['half']
            if id(b) not in floor:
                counted_vol += 8.0 * hx * hy * hz
            m = b['mass']
            mass += m
            moment += m * b['center'][2]
            boxes.append((cs, b))
    if not boxes:
        return 0.0
    fill = min(100.0 * counted_vol / denom, 100.0)
    cog = (max(0.0, min(100.0, 100.0 * (1.0 - (moment / mass) / height)))
           if mass > 0.0 and height > 0.0 else 0.0)
    return (fill + cog
            + _class_score(boxes, 'is_prioritized', any_prio_c)
            + _class_score(boxes, 'is_soft', False)) / 4.0


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
            # Deferral features. placement_score and soft_item_score penalise
            # a priority/soft item with *anything* of another class above it
            # in its column -- not merely resting on it -- so those items are
            # only safe near the top of the load. choose_action is otherwise
            # completely class-blind.
            if W_SOFT_DEFER and spec.get('is_soft'):
                score -= W_SOFT_DEFER
            if W_PRIO_DEFER and spec.get('is_prioritized'):
                score -= W_PRIO_DEFER
            if score > best_score:
                best_score = score
                best = (pool_idx, cidx, pos, orn)
    return best
