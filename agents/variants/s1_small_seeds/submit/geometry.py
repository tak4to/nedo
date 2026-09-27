"""Geometric feasibility oracle.

This module re-derives, in pure Python/float math, the exact same checks
that ``src/ground_handling/validator.py`` performs on the evaluation
server (inclusion inside the container, and the "push in from the door"
transport-path collision check), so the agent can validate a candidate
placement *before* ever submitting it as an action.

Everything here works in a container's LOCAL frame, i.e. relative to
``(offset_x, 0, 0)`` -- the same frame ``policy()`` must return
``place_pos`` in. All dimensions are derived at runtime from the
``container_list`` observation dict; nothing about container size is
hard-coded, so this works for any container configuration the evaluator
throws at us (1 or 2 containers, shelf or no shelf, any L/W/H/cut_x/cut_y).

Values below marked "documented eval constant" come straight from the
README's evaluation-infra parameter table. We additionally pad every
tolerance a bit beyond that documented minimum, since (a) we can't
observe the validator's own config at runtime, and (b) placements settle
by 1-2cm after being warped in, so relying on the bare minimum is fragile.
"""
import math

# ---- documented eval-infra constants (README) ----
SAFETY_MARGIN = 0.015        # collision distance threshold during transport
INCLUSION_MARGIN = -0.005    # must be <= this on every wall plane (negative = inside)
CEILING_MARGIN = 0.018
START_Z = 0.08
SETTLE_DISPLACEMENT_THRESH = 0.3
SETTLE_ANGLE_THRESH_DEG = 45.0

# ---- undocumented default (BaseValidator.start_margin) ----
START_MARGIN = 0.01

# ---- our own extra safety padding on top of the above ----
GAP = 0.022                  # design clearance between neighboring items (> SAFETY_MARGIN)
WALL_CLEARANCE = 0.022       # design clearance from container walls (> -EFFECTIVE_INCLUSION_MARGIN)
BAND_PUSH = 0.06             # jump used to clear the "resting surface" danger band
EPS = 1e-6

# Flat extra padding applied on top of the documented INCLUSION_MARGIN
# for every plane check (independent of WALL_CLEARANCE, which is a
# candidate-generation concern, not the pass/fail threshold itself).
INCLUSION_SAFETY_PAD = 0.01
EFFECTIVE_INCLUSION_MARGIN = INCLUSION_MARGIN - INCLUSION_SAFETY_PAD  # -0.015

# The container floor is *both* one of the 7 inclusion planes and the
# transport check's "resting surface" -- resting exactly flush (gap=0)
# satisfies the transport logic but leaves zero slack on the inclusion
# check (which needs clearance from every plane, floor included). Must
# exceed -EFFECTIVE_INCLUSION_MARGIN. Stacking on top of another *item*
# has no such issue (inclusion only checks the outer container boundary,
# not other items), so this lift is only needed for true floor placements.
FLOOR_LIFT = 0.02

# orn_idx -> which of (l/2, w/2, h/2) maps to (X, Y, Z) half-extent.
# Matches src/ground_handling/utils.py:get_half_ext exactly.
_ORN_HALF_IDX = {
    0: (0, 1, 2),
    1: (0, 2, 1),
    2: (2, 1, 0),
    3: (1, 0, 2),
    4: (1, 2, 0),
    5: (2, 0, 1),
}
ALL_ORNS = (0, 1, 2, 3, 4, 5)


def half_extents(length, width, height, orn_idx):
    lwh_half = (length / 2.0, width / 2.0, height / 2.0)
    ia, ib, ic = _ORN_HALF_IDX[orn_idx]
    return (lwh_half[ia], lwh_half[ib], lwh_half[ic])


def quat_to_matrix(quat):
    """(x,y,z,w) quaternion -> row-major 3x3 rotation matrix (tuple of 9)."""
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    )


def aabb_half_extents_from_quat(length, width, height, quat):
    """World-axis-aligned half-extents of a box rotated by an (assumed
    axis-aligned, but possibly drifted) quaternion. General formula
    (sum of |R_ij| * local half-extent), robust to any small numerical
    drift from the physics settling step."""
    m = quat_to_matrix(quat)
    hl = (length / 2.0, width / 2.0, height / 2.0)
    hx = abs(m[0]) * hl[0] + abs(m[1]) * hl[1] + abs(m[2]) * hl[2]
    hy = abs(m[3]) * hl[0] + abs(m[4]) * hl[1] + abs(m[5]) * hl[2]
    hz = abs(m[6]) * hl[0] + abs(m[7]) * hl[1] + abs(m[8]) * hl[2]
    return (hx, hy, hz)


def container_geometry(cdict):
    """Derive all geometric facts about a container from the observation
    dict alone (container_list[i]). Returns a plain dict used by every
    other function in this module."""
    length = cdict['length']; width = cdict['width']; height = cdict['height']
    thickness = cdict['thickness']; cut_x = cdict['cut_x']; cut_y = cdict['cut_y']
    center = cdict['center']
    offset_x = center[0]
    # container.center z = height/2 + buffer (see containers.py Container.create)
    buffer = center[2] - height / 2.0

    points_local = [(p[0] - offset_x, p[1], p[2]) for p in cdict['points']]
    n_vecs = [tuple(n) for n in cdict['n_vecs']]

    # The one non-axis-aligned plane among the 7: write_open_cut_corner_cup_obj
    # (src/ground_handling/utils.py) extrudes a pentagon cross-section (5 side
    # planes + 2 Y end-caps = 7), and exactly one of the 5 side planes -- the
    # chamfer joining the floor to the left wall -- has a normal with both a
    # non-zero X and non-zero Z component (the other 4 side planes are each
    # purely X or purely Z, and the 2 end-caps are purely Y). None if the
    # container has no such plane (shouldn't happen given cut_x/cut_y are
    # always > 0, but this is a geometric fact worth deriving, not assuming).
    diag_n = diag_p = None
    for n, p in zip(n_vecs, points_local):
        if abs(n[1]) < 1e-6 and abs(n[0]) > 1e-6 and abs(n[2]) > 1e-6:
            diag_n, diag_p = n, p
            break

    floor_struct_z = thickness
    shelf_struct_z = height / 2.0 + thickness + buffer

    small_shelf_center = (-(length / 2.0 - cut_x / 2.0 - thickness), 0.0,
                           height / 2.0 + thickness / 2.0 + buffer)
    small_shelf_half = (cut_x / 2.0, width / 2.0 - thickness, thickness / 2.0)
    static_boxes = [(small_shelf_center, small_shelf_half)]

    if cdict.get('shelf'):
        shelf_center = (0.0, width / 4.0, height / 2.0 + thickness / 2.0 + buffer)
        shelf_half = (length / 2.0 - thickness / 2.0, width / 4.0 - thickness, thickness / 2.0)
        static_boxes.append((shelf_center, shelf_half))

    return {
        'index': cdict['index'],
        'length': length, 'width': width, 'height': height,
        'thickness': thickness, 'cut_x': cut_x, 'cut_y': cut_y,
        'buffer': buffer, 'offset_x': offset_x,
        'is_prioritized': bool(cdict.get('is_prioritized', False)),
        'has_shelf': bool(cdict.get('shelf', False)),
        'points_local': points_local, 'n_vecs': n_vecs,
        'diag_n': diag_n, 'diag_p': diag_p,
        'floor_struct_z': floor_struct_z, 'shelf_struct_z': shelf_struct_z,
        'static_boxes': static_boxes,
        # Usable design box, mirroring the 7 real inclusion planes with
        # WALL_CLEARANCE of slack. These are only *candidate-generation*
        # bounds -- check_inclusion stays the authority, and it is what
        # actually enforces the diagonal chamfer at the bottom-left.
        #
        # x_lo spans the full inner width including the notch strip: that
        # strip is reachable (the validator enters at the door then slides
        # sideways in X) and is ~23% of the container, so excluding it
        # outright forfeits far more than it protects against.
        # y_lo has NO thickness term: the door side is an opening, not a
        # wall, so the front inclusion plane sits at -width/2 exactly.
        'x_lo': -length / 2.0 + thickness + WALL_CLEARANCE,
        'x_hi': length / 2.0 - thickness - WALL_CLEARANCE,
        'y_lo': -width / 2.0 + WALL_CLEARANCE,
        'y_hi': width / 2.0 - thickness - WALL_CLEARANCE,
        'z_lo': thickness + buffer + WALL_CLEARANCE,
        'z_hi': height + buffer - thickness - WALL_CLEARANCE,
    }


def check_inclusion(cgeom, pos_local, half_ext, margin=EFFECTIVE_INCLUSION_MARGIN):
    px, py, pz = pos_local
    hx, hy, hz = half_ext
    for n, p in zip(cgeom['n_vecs'], cgeom['points_local']):
        dot = n[0] * (px - p[0]) + n[1] * (py - p[1]) + n[2] * (pz - p[2])
        dot += abs(n[0]) * hx + abs(n[1]) * hy + abs(n[2]) * hz
        if dot > margin:
            return False
    return True


def _aabb_distance(c1, h1, c2, h2):
    gaps = []
    for ax in range(3):
        gap = max(c1[ax] - h1[ax] - (c2[ax] + h2[ax]), c2[ax] - h2[ax] - (c1[ax] + h1[ax]), 0.0)
        gaps.append(gap)
    return math.sqrt(sum(g * g for g in gaps))



def _swept_clear(obstacles, axis, box_center, box_half, v_from, v_to, margin):
    lo, hi = (v_from, v_to) if v_from <= v_to else (v_to, v_from)
    lo -= box_half[axis]
    hi += box_half[axis]
    for (oc, oh) in obstacles:
        gaps = []
        for ax in range(3):
            if ax == axis:
                gap = max(oc[ax] - oh[ax] - hi, lo - (oc[ax] + oh[ax]), 0.0)
            else:
                gap = max(abs(box_center[ax] - oc[ax]) - box_half[ax] - oh[ax], 0.0)
            gaps.append(gap)
        if math.sqrt(sum(g * g for g in gaps)) <= margin:
            return False
    return True


def resting_eff_start_z(cgeom, target_z, half_z):
    resting = (cgeom['floor_struct_z'], cgeom['shelf_struct_z'])
    ceiling = (cgeom['height'] / 2.0 + cgeom['buffer'],
               cgeom['height'] + cgeom['buffer'] - cgeom['thickness'])
    eff = START_Z
    bottom_z = target_z - half_z
    for r in resting:
        if 0.0 <= (bottom_z - r) <= 0.05:
            eff = 0.0
            break
    top_z = target_z + half_z
    if eff > 0.0:
        for c in ceiling:
            clearance = c - top_z
            if 0.0 <= clearance < (eff + CEILING_MARGIN):
                eff = max(0.0, clearance - CEILING_MARGIN - 0.0005)
                break
    return eff


def x_transport_range(cgeom, half_x):
    x_min = -cgeom['length'] / 2.0 + cgeom['thickness'] + cgeom['cut_x'] + half_x + START_MARGIN
    x_max = cgeom['length'] / 2.0 - cgeom['thickness'] - half_x - START_MARGIN
    return x_min, x_max


def check_transport_path(cgeom, obstacle_boxes, target_local, half_ext, safety_margin=None):
    """obstacle_boxes: list of (center_xyz, half_xyz) tuples in local coords
    (already-packed items' AABBs + the container's static shelf boxes)."""
    if safety_margin is None:
        safety_margin = GAP
    hx, hy, hz = half_ext
    x_min, x_max = x_transport_range(cgeom, hx)
    rel_x = min(max(target_local[0], x_min), x_max)

    eff = resting_eff_start_z(cgeom, target_local[2], hz)
    z_cap = cgeom['height'] + cgeom['buffer'] - cgeom['thickness'] - hz - START_MARGIN
    rel_z = min(z_cap, target_local[2] + eff)

    start_y = -cgeom['width'] / 2.0

    box_center = (rel_x, start_y, rel_z)
    if not _swept_clear(obstacle_boxes, 1, box_center, half_ext, start_y, target_local[1], safety_margin):
        return False

    box_center2 = (rel_x, target_local[1], rel_z)
    if not _swept_clear(obstacle_boxes, 0, box_center2, half_ext, rel_x, target_local[0], safety_margin):
        return False

    return True



def validate_placement(cgeom, obstacle_boxes, target_local, half_ext):
    """Full oracle: True iff this placement is both included and reachable."""
    if not check_inclusion(cgeom, target_local, half_ext):
        return False
    if not check_transport_path(cgeom, obstacle_boxes, target_local, half_ext):
        return False
    return True


def diagonal_corner_x(cgeom, cz, hx, hz, pad=WALL_CLEARANCE):
    """The 'moving extreme point' for the container's diagonal corner
    chamfer (docs/2026-09-20-strategy.md A-5): how far left an item's left
    edge can sit at height `cz` while still honouring the chamfer plane,
    with the same WALL_CLEARANCE margin every other candidate-generation
    bound in container_geometry uses.

    x_lo (the plain axis-aligned left-wall bound) is a valid but often
    needlessly conservative stand-in for this near the floor: the chamfer
    only clips the bottom-left corner, so an item low and short enough to
    clear it can sit further left than x_lo alone would ever propose,
    leaving a wedge of usable volume near the chamfer completely
    unreachable by the ordinary flush-against-x_lo extreme points. This
    derives the tight bound directly from the plane check_inclusion
    itself uses (cgeom['diag_n']/['diag_p']), not from re-deriving
    cut_x/cut_y by hand, so it can't drift out of sync with the actual
    inclusion oracle.

    Returns cgeom['x_lo'] unchanged (a no-op) when the container has no
    such plane, or when the chamfer isn't binding at this height/size (the
    axis-aligned wall is then the tighter bound, e.g. cz above the
    chamfer's z-range).
    """
    n, p = cgeom.get('diag_n'), cgeom.get('diag_p')
    if n is None:
        return cgeom['x_lo']
    nx, _ny, nz = n
    px, _py, pz = p
    margin = -pad
    rhs = margin - nz * (cz - pz) - abs(nx) * hx - abs(nz) * hz + nx * px
    cx_bound = rhs / nx
    return max(cgeom['x_lo'], cx_bound - hx)
