"""Where does the empty volume sit when an episode ends stuck?

Voxelises the final container (2cm) and splits every empty voxel into:
  enclosed   below the column's skyline (under an overhang / between stacked items)
  ceil_dead  above the skyline, but the air column is shorter than the
             smallest remaining item dimension + lift -> nothing can ever stand there
  open_air   above the skyline and tall enough -- unusable only because no
             footprint fits (slivers) or the door route is blocked
Also reports how much of open_air sits in columns whose free width (the
run of neighbouring cells at a similar skyline) is under the smallest
remaining footprint side.

    ../.venv/bin/python void_budget.py P15 P04 P10
"""
import sys, json, os
ROOT = '/home/takato/comp/nedo'
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, os.path.join(ROOT, 'agents', 'submit'))
import numpy as np
from diag_stuck import run
import packer, geometry

CELL = 0.02


def analyse(nm, scenes='scenes_pool.json'):
    cfg = json.load(open(os.path.join(ROOT, 'tools', scenes)))[nm]
    cfg['containers']['container_list'] = cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers'] = 1
    env, obs, last, steps = run(nm, cfg, opt=15.0)
    cd = env.container_manager.get_item_info_in_containers()[0]
    cs = packer.ContainerState(cd); g = cs.geom
    rem = [i for i in env.stream_manager.visible_pool] + list(getattr(env.stream_manager, 'item_queue', []) or [])
    dims = sorted({round(d, 3) for i in rem for d in (i.length, i.width, i.height)}) or [0.2]
    dmin = dims[0]
    xs = np.arange(g['x_lo'] + CELL / 2, g['x_hi'], CELL)
    ys = np.arange(g['y_lo'] + CELL / 2, g['y_hi'], CELL)
    zs = np.arange(g['z_lo'] + CELL / 2, g['z_hi'], CELL)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    occ = np.zeros(X.shape, bool); ost = np.zeros(X.shape, bool)
    for b in cs.boxes:
        (cx, cy, cz), (hx, hy, hz) = b['center'], b['half']
        occ |= ((np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy) & (np.abs(Z - cz) <= hz))
    for b in cs.static_obstacles:
        (cx, cy, cz), (hx, hy, hz) = b['center'], b['half']
        ost |= ((np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy) & (np.abs(Z - cz) <= hz))
    # inside the container (7 planes incl. the chamfer)
    inside = np.ones(X.shape, bool)
    for n, p in zip(g['n_vecs'], g['points_local']):
        inside &= (n[0] * (X - p[0]) + n[1] * (Y - p[1]) + n[2] * (Z - p[2])) <= 0
    # skyline index per column = highest occupied voxel
    zidx = np.arange(len(zs))[None, None, :]
    top = np.where(occ, zidx, -1).max(axis=2)            # item skyline, -1 = no item in column
    below = zidx <= top[:, :, None]
    empty = inside & ~occ & ~ost
    # enclosed = first thing straight above is an item (under a static shelf is reachable)
    lab = np.zeros(occ.shape[:2], np.int8); enc = np.zeros(occ.shape, bool)
    for k in range(occ.shape[2] - 1, -1, -1):
        enc[:, :, k] = ~occ[:, :, k] & ~ost[:, :, k] & (lab == 1)
        lab = np.where(occ[:, :, k], 1, np.where(ost[:, :, k], 2, lab))
    enclosed = empty & enc
    air = empty & ~enc
    air_h = (air & ~below).sum(axis=2) * CELL             # air above the item skyline
    need = dmin + packer.SUPPORT_LIFT
    ceil_dead_col = air_h < need
    above = air & ~below
    ceil_dead = above & ceil_dead_col[:, :, None]
    open_air = above & ~ceil_dead_col[:, :, None]
    # free-width proxy: run length along x and along y of columns whose skyline
    # is within 3cm of this one (a footprint needs both runs >= dmin + 2*GAP)
    sky = np.where(top >= 0, (top + 1) * CELL, 0.0)
    need_w = dmin + 2 * geometry.GAP
    def run_len(a, axis):
        out = np.zeros_like(a, dtype=float)
        n = a.shape[axis]
        for i in range(n):
            sl = [slice(None)] * 2; sl[axis] = i
            base = a[tuple(sl)]
            L = np.ones_like(base, dtype=float)
            for d in (-1, 1):
                j = i + d; alive = np.ones_like(base, bool)
                while 0 <= j < n:
                    sl2 = [slice(None)] * 2; sl2[axis] = j
                    alive &= np.abs(a[tuple(sl2)] - base) <= 0.03
                    if not alive.any(): break
                    L += alive; j += d
            out[tuple(sl)] = L * CELL
        return out
    rx = run_len(sky, 0); ry = run_len(sky, 1)
    narrow_col = (np.minimum(rx, ry) < need_w) & ~ceil_dead_col
    sliver = open_air & narrow_col[:, :, None]
    wide = open_air & ~narrow_col[:, :, None]
    V = inside.sum()
    f = lambda m: 100.0 * m.sum() / V
    # sealed: some item voxel lies between the door (y_lo) and this voxel at the
    # same (x, z). A straight +y push at this height cannot get here even with a
    # zero-width item, so this is a LOWER bound on unreachable air.
    blocked = np.cumsum(occ, axis=1) > 0          # any item at or in front of this y
    sealed_air = air & ~below & blocked
    reach_air = air & ~below & ~blocked
    items = 100.0 * (occ & inside).sum() / V
    under_shelf = f(air & below)
    print(f"{nm:5s} items={items:5.1f}%  under_static={under_shelf:5.1f}%  enclosed={f(enclosed):5.1f}%  ceil_dead={f(ceil_dead):5.1f}%  "
          f"sliver={f(sliver):5.1f}%  wide_open={f(wide):5.1f}%  | above-sky air: sealed={f(sealed_air):5.1f}% reachable={f(reach_air):5.1f}%   (dmin={dmin:.2f}, need_h={need:.2f}, need_w={need_w:.3f})")
    return dict(scene=nm, items=items, under_static=under_shelf, sealed=f(sealed_air), reach=f(reach_air), enclosed=f(enclosed), ceil_dead=f(ceil_dead), sliver=f(sliver), wide=f(wide))


if __name__ == '__main__':
    rows = [analyse(n) for n in sys.argv[1:]]
    if len(rows) > 1:
        k = ['items', 'sealed', 'reach', 'under_static', 'enclosed', 'ceil_dead', 'sliver', 'wide']
        print('mean  ' + '  '.join(f"{x}={np.mean([r[x] for r in rows]):5.1f}%" for x in k))
