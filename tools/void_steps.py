"""Which placements create the permanently dead volume?

Replays one episode and, after every placement, recomputes (2cm voxels):
  enclosed  empty voxels below the column skyline
  shadow    the part of `enclosed` directly under the NEW item's footprint
            (overhang: air sealed beneath it) -- attributed to that step
Prints per-step deltas with the placement's features, then totals by
feature buckets.

    ../.venv/bin/python void_steps.py P15 P04
"""
import sys, json, os, io, contextlib, collections
ROOT = '/home/takato/comp/nedo'
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'agents', 'submit'))
import numpy as np
from src.ground_handling.env import GroundHandlingEnv
import agent as agent_module, packer

CELL = 0.02


class Vox:
    def __init__(self, g):
        self.g = g
        self.xs = np.arange(g['x_lo'] + CELL / 2, g['x_hi'], CELL)
        self.ys = np.arange(g['y_lo'] + CELL / 2, g['y_hi'], CELL)
        self.zs = np.arange(g['z_lo'] + CELL / 2, g['z_hi'], CELL)
        X, Y, Z = np.meshgrid(self.xs, self.ys, self.zs, indexing='ij')
        self.X, self.Y, self.Z = X, Y, Z
        ins = np.ones(X.shape, bool)
        for n, p in zip(g['n_vecs'], g['points_local']):
            ins &= (n[0] * (X - p[0]) + n[1] * (Y - p[1]) + n[2] * (Z - p[2])) <= 0
        self.inside = ins
        self.V = ins.sum()

    def occ(self, boxes):
        o = np.zeros(self.X.shape, bool)
        for (cx, cy, cz), (hx, hy, hz) in boxes:
            ix = np.abs(self.xs - cx) <= hx; iy = np.abs(self.ys - cy) <= hy; iz = np.abs(self.zs - cz) <= hz
            o[np.ix_(ix, iy, iz)] = True
        return o

    def enclosed(self, o_item, o_static):
        """Empty voxels whose first occupied voxel straight above is an ITEM.
        Space under a static shelf is not enclosed -- it is reachable by
        the under-shelf candidates."""
        lab = np.zeros(o_item.shape[:2], np.int8)   # 0 none, 1 item, 2 static
        out = np.zeros(o_item.shape, bool)
        for k in range(o_item.shape[2] - 1, -1, -1):
            empty = ~o_item[:, :, k] & ~o_static[:, :, k]
            out[:, :, k] = empty & (lab == 1)
            lab = np.where(o_item[:, :, k], 1, np.where(o_static[:, :, k], 2, lab))
        return out & self.inside


def run(nm, scenes='scenes_pool.json'):
    cfg = json.load(open(os.path.join(ROOT, 'tools', scenes)))[nm]
    agent_module.OPTIMIZE_TIME_BUDGET = 15.0
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False)
        ag = agent_module.Agent(module_path=os.path.join(ROOT, 'agents', 'submit'))
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream(); obs, _ = env.reset(seed=42)
        vox = {}; prev = {}
        term = False; steps = 0
        while not term and steps < 500:
            pool = obs['pool_list']
            a = ag.policy(observation=obs)
            it = pool[a['item_idx']]
            obs, r, term, tr, info = env.step(a); steps += 1
            st = info.get('status', {})
            if st.get('is_placed_safe') is False or st.get('is_valid') is False:
                break
            ci = a['container_idx']
            cd = env.container_manager.get_item_info_in_containers()[ci]
            cs = packer.ContainerState(cd)
            if ci not in vox:
                vox[ci] = Vox(cs.geom); prev[ci] = 0
            V = vox[ci]
            oi = V.occ([(b['center'], b['half']) for b in cs.boxes])
            osx = V.occ([(b['center'], b['half']) for b in cs.static_obstacles])
            enc = V.enclosed(oi, osx)
            e = int(enc.sum())
            # the box that was just placed, found by item index (packed_items order is not placement order)
            pi = next(p for p in cd['packed_items'] if p['index'] == it['index'])
            hx, hy, hz = packer.aabb_half_extents_from_quat(pi['length'], pi['width'], pi['height'], pi['orn'])
            cx, cy, cz = pi['pos'][0] - cs.geom['offset_x'], pi['pos'][1], pi['pos'][2]
            ix = np.abs(V.xs - cx) <= hx; iy = np.abs(V.ys - cy) <= hy; iz = V.zs < cz - hz
            sh = int(enc[np.ix_(ix, iy, iz)].sum())
            d = e - prev[ci]; prev[ci] = e
            top, sup = packer._landing(packer.ContainerState(cd), cx, cy, hx, hy)
            rows.append(dict(scene=nm, step=steps, c=ci, dims=(it['length'], it['width'], it['height']),
                             fz=2 * hz, bottom=cz - hz, z_rel=(cz - hz - cs.geom['z_lo']) / (cs.geom['z_hi'] - cs.geom['z_lo']),
                             d_enc=100.0 * d / V.V, shadow=100.0 * sh / V.V,
                             floor=(cz - hz - cs.geom['z_lo']) < 0.05))
        env.close()
    return rows


if __name__ == '__main__':
    allr = []
    for n in sys.argv[1:]:
        rows = run(n); allr += rows
        tot = sum(r['d_enc'] for r in rows); sh = sum(r['shadow'] for r in rows)
        print(f"{n}: {len(rows)} placed, enclosed total {tot:5.1f}% of container, of which shadow-under-new-item {sh:5.1f}%")
        big = sorted(rows, key=lambda r: -r['d_enc'])[:5]
        for r in big:
            print(f"   step {r['step']:2d} dims={r['dims']} fz={r['fz']:.2f} bottom={r['bottom']:.2f} "
                  f"d_enc={r['d_enc']:+5.2f}% shadow={r['shadow']:4.2f}%")
    print("\nby layer (bottom height as fraction of usable height):")
    b = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in allr:
        k = 'floor' if r['floor'] else f"{min(int(r['z_rel'] * 4), 3) * 25}-{min(int(r['z_rel'] * 4), 3) * 25 + 25}%"
        b[k][0] += 1; b[k][1] += r['d_enc']; b[k][2] += r['shadow']
    for k in sorted(b):
        n, d, s = b[k]
        print(f"   {k:8s} n={n:3d}  enclosed {d:6.1f}%  (per item {d / n:4.2f}%)  shadow {s:5.1f}%")
