"""Face-to-face horizontal gaps at the end of an episode.

For every placed box and each horizontal direction (+x, -x, +y back), find
the nearest obstacle face (another box overlapping in the other horizontal
axis AND in z, or a container wall). Classify the gap:
  tight   <= DEAD_LO   (the designed clearance, unavoidable)
  dead    (DEAD_LO, DEAD_HI)  -- too narrow for ANY catalogue item + clearance
  usable  >= DEAD_HI
and weight by the facing area (so the output is dead volume, % of container).
-y (the door side) is skipped: the space in front of an item is open.

    ../.venv/bin/python dead_gaps.py P15 P04
"""
import sys, json, os, collections
ROOT = '/home/takato/comp/nedo'
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, os.path.join(ROOT, 'agents', 'submit'))
import numpy as np
from diag_stuck import run
import packer, geometry

CATALOG_MIN = 0.20                       # smallest dimension of any item type
DEAD_LO = geometry.GAP + 0.012           # 0.034: anything tighter is designed clearance
DEAD_HI = CATALOG_MIN + 2 * geometry.GAP  # 0.244: narrower than this fits nothing, ever


def gaps(cs):
    g = cs.geom
    B = [(b['center'], b['half']) for b in cs.boxes]
    S = [(b['center'], b['half']) for b in cs.static_obstacles]
    out = []
    for i, ((cx, cy, cz), (hx, hy, hz)) in enumerate(B):
        for ax, sgn in ((0, 1), (0, -1), (1, 1)):
            face = (cx, cy)[ax] + sgn * (hx, hy)[ax]
            wall = (g['x_hi'], g['x_lo'], g['y_hi'])[(0 if sgn > 0 else 1) if ax == 0 else 2]
            best = abs(wall - face) + geometry.WALL_CLEARANCE
            other = 1 - ax
            oc, oh = (cx, cy)[other], (hx, hy)[other]
            for j, ((dx, dy, dz), (ex, ey, ez)) in enumerate(B + S):
                if j == i:
                    continue
                # overlap in the other horizontal axis and in z
                if abs((dx, dy)[other] - oc) >= (ex, ey)[other] + oh - 1e-3:
                    continue
                if abs(dz - cz) >= ez + hz - 1e-3:
                    continue
                f2 = (dx, dy)[ax] - sgn * (ex, ey)[ax]
                d = (f2 - face) * sgn
                if d >= -1e-3:
                    best = min(best, max(d, 0.0))
            area = (2 * (hy, hx)[ax]) * 2 * hz    # facing area: other-axis extent x height
            out.append((best, area, 'x' if ax == 0 else 'y'))
    return out


def main():
    vol_tot = collections.Counter(); n_tot = collections.Counter()
    for nm in sys.argv[1:]:
        cfg = json.load(open(os.path.join(ROOT, 'tools', 'scenes_pool.json')))[nm]
        cfg['containers']['container_list'] = cfg['containers']['container_list'][:1]
        cfg['camera']['num_containers'] = 1
        env, obs, last, steps = run(nm, cfg, opt=15.0)
        cd = env.container_manager.get_item_info_in_containers()[0]
        cs = packer.ContainerState(cd)
        V = env.container_manager.containers[0].volume
        vol = collections.Counter(); cnt = collections.Counter()
        for gap, area, dirn in gaps(cs):
            k = 'tight' if gap <= DEAD_LO else ('dead' if gap < DEAD_HI else 'usable')
            cnt[k] += 1
            if k == 'dead':
                vol['dead_' + dirn] += gap * area
            # a gap is shared by two faces when both sides are boxes; halve the
            # volume for box-box gaps is overkill here -- report per face, upper bound
            if k == 'dead':
                vol[k] += gap * area
        print(f"{nm}: faces tight={cnt['tight']:3d} dead={cnt['dead']:3d} usable={cnt['usable']:3d}  "
              f"dead-gap volume <= {100 * vol['dead'] / V:4.1f}% of container  (x-dir {100 * vol['dead_x'] / V:4.1f}%, y-dir {100 * vol['dead_y'] / V:4.1f}%)")
        vol_tot['x'] += 100 * vol['dead_x'] / V; vol_tot['y'] += 100 * vol['dead_y'] / V
        for k in cnt: n_tot[k] += cnt[k]
        vol_tot['dead'] += 100 * vol['dead'] / V
    n = len(sys.argv) - 1
    print(f"mean dead faces {n_tot['dead'] / n:.1f}/scene, dead-gap volume <= {vol_tot['dead'] / n:.1f}%  "
          f"x-dir {vol_tot['x'] / n:.1f}%  y-dir {vol_tot['y'] / n:.1f}%  (dead zone {DEAD_LO:.3f}-{DEAD_HI:.3f} m)")


if __name__ == '__main__':
    main()
