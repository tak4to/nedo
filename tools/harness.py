"""In-process harness: runs an agent through GroundHandlingEnv and reports
fill_score plus the four scores the local evaluator does NOT implement
(cog / placement / soft / stability proxies)."""
import os, sys, json, time, io, contextlib, math, importlib, argparse
import numpy as np

ROOT = '/home/takato/comp/nedo'
# The agent under test. Set AGENT_DIR to A/B two agent directories --
# importing this module used to unconditionally push agents/submit to the
# front of sys.path, which silently overrode any directory the caller had
# already put there and made every A/B compare an agent against itself.
AGENT_DIR = os.environ.get('AGENT_DIR') or os.path.join(ROOT, 'agents', 'submit')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from src.ground_handling.env import GroundHandlingEnv  # noqa


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def fill_with_margin(env, margin):
    """Recompute fill_score with an arbitrary inclusion margin."""
    total_v = 0.0
    denom = sum(c.volume for c in env.container_manager.containers)
    n_in = 0
    reasons = {}
    for container in env.container_manager.containers:
        n_vecs = np.array(container.n_vecs); points = np.array(container.points)
        for item in container.packed_items:
            pos, orn = item.get_pose(env.client)
            if pos is None:
                continue
            R = np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            hl, hw, hh = item.length/2, item.width/2, item.height/2
            lc = np.array([[sx*hl, sy*hw, sz*hh] for sx in (1,-1) for sy in (1,-1) for sz in (1,-1)])
            gc = lc @ R.T + np.array(pos)
            ok = True
            for corner in gc:
                dots = np.sum(n_vecs * (corner - points), axis=1)
                if np.any(dots > margin):
                    ok = False
                    reasons[int(np.argmax(dots))] = reasons.get(int(np.argmax(dots)), 0) + 1
                    break
            if ok:
                total_v += item.volume
                n_in += 1
    return min(100 * total_v / denom, 100), n_in, reasons


def extra_scores(env):
    """Proxies for cog / placement / soft. Not the official formulas
    (unknown) but monotone in the same direction."""
    out = {}
    conts = env.container_manager.containers
    tot_m = 0.0; mz = 0.0; hmax = 0.0
    boxes = []
    for c in conts:
        hmax = max(hmax, c.height)
        for it in c.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            R = np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            hl = np.abs(R) @ np.array([it.length/2, it.width/2, it.height/2])
            boxes.append(dict(pos=np.array(pos), h=hl, it=it, c=c))
            tot_m += it.mass; mz += it.mass * pos[2]
    if tot_m == 0:
        return dict(cog_h=0.0, cog_score=0.0, n_prio_buried=0, n_soft_buried=0,
                    n_prio_wrong=0, top_z=0.0)
    cog_z = mz / tot_m
    # normalized: 0 at ceiling, 100 at floor
    cog_score = max(0.0, min(100.0, 100 * (1 - cog_z / hmax)))
    # "buried": another item of a different class sits directly above
    n_prio_buried = n_soft_buried = 0
    for b in boxes:
        it = b['it']
        if not (it.is_prioritized or it.is_soft):
            continue
        top = b['pos'][2] + b['h'][2]
        covered = False
        for o in boxes:
            if o is b or o['c'] is not b['c']:
                continue
            oi = o['it']
            same = (oi.is_prioritized == it.is_prioritized) and (oi.is_soft == it.is_soft)
            if same:
                continue
            if o['pos'][2] - o['h'][2] < top - 0.02:
                continue
            if (abs(o['pos'][0]-b['pos'][0]) < o['h'][0]+b['h'][0] and
                    abs(o['pos'][1]-b['pos'][1]) < o['h'][1]+b['h'][1]):
                covered = True; break
        if covered:
            if it.is_prioritized: n_prio_buried += 1
            if it.is_soft: n_soft_buried += 1
    any_prio_c = any(c.is_prioritized for c in conts)
    n_prio_wrong = 0
    if any_prio_c:
        for b in boxes:
            if b['it'].is_prioritized and not b['c'].is_prioritized:
                n_prio_wrong += 1
    top_z = max((b['pos'][2] + b['h'][2]) for b in boxes)
    return dict(cog_h=cog_z, cog_score=cog_score, n_prio_buried=n_prio_buried,
                n_soft_buried=n_soft_buried, n_prio_wrong=n_prio_wrong, top_z=top_z)


def ceiling_fill(cfg):
    """fill_score if every item were counted."""
    from src.ground_handling.containers import Container
    tot_v = sum(i['length']*i['width']*i['height'] for i in cfg['item_stream']['item_list'])
    denom = 0.0
    for cd in cfg['containers']['container_list']:
        L, W, H, t = cd['length'], cd['width'], cd['height'], cd['thickness']
        il, iw, ih = L-2*t, W-2*t, H-2*t
        base = il*iw*ih
        cut = 0.5*(cd['cut_x']-t)*(cd['cut_y']-t)*iw
        ss = cd['cut_x']*t*iw
        sh = il*t*(iw/2) if cd.get('require_shelf') else 0.0
        denom += base - cut - ss - sh
    return min(100*tot_v/denom, 100)


# Constants to force onto the *agent* module, applied after the reload below.
# geometry/packer constants can be set by the caller before run_scene, but
# agent's cannot: run_scene reloads that module, which would wipe them. Set
# this dict instead (ab.py --set a:NAME=value does).
AGENT_OVERRIDES = {}


def run_scene(name, cfg, agent_mod='agent', policy_budget=None, optimize_budget=None):
    import agent as agent_module
    importlib.reload(agent_module)
    for k, v in AGENT_OVERRIDES.items():
        setattr(agent_module, k, v)
    if policy_budget is not None:
        agent_module.POLICY_TIME_BUDGET = policy_budget
    if optimize_budget is not None:
        agent_module.OPTIMIZE_TIME_BUDGET = optimize_budget

    t0 = time.perf_counter()
    with quiet():
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = agent_module.Agent(module_path=os.path.join(ROOT, 'agents', 'submit'))
        env.reset_settings()
        init = env.get_init_states()
        ag.get_init_states(init)
        if cfg['agent']['optimize']:
            order = ag.optimize(env.get_info_for_optimization())
            env.set_item_order(order)
        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        steps = 0
        max_pol = 0.0
        term = False
        last_info = {}
        while not term and steps < 500:
            tp = time.perf_counter()
            action = ag.policy(observation=obs)
            max_pol = max(max_pol, time.perf_counter() - tp)
            obs, r, term, trunc, last_info = env.step(action)
            steps += 1
        rep = env.evaluate()
        f_strict, n_strict, reasons = fill_with_margin(env, -0.005)
        f_loose, n_loose, _ = fill_with_margin(env, 0.02)
        ex = extra_scores(env)
        try:
            import scores
            ex.update(scores.all_scores(env))
        except Exception:
            import traceback; traceback.print_exc()
        n_packed = sum(len(c.packed_items) for c in env.container_manager.containers)
        tot = env.num_total_items
        env.close()
    ceil_ = ceiling_fill(cfg)
    return dict(scene=name, fill=rep['fill_score'], ceiling=ceil_,
                norm=100*rep['fill_score']/ceil_ if ceil_ else 0,
                fill_loose=f_loose, n_in_strict=n_strict, n_in_loose=n_loose,
                packed=n_packed, total=tot, pct=100*n_packed/tot,
                steps=steps, max_policy_s=max_pol, wall_s=time.perf_counter()-t0,
                status=last_info.get('status'), **ex)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes.json')
    ap.add_argument('--only', default=None)
    ap.add_argument('--policy-budget', type=float, default=None)
    ap.add_argument('--optimize-budget', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    scenes = json.load(open(a.scenes))
    if a.only:
        keys = a.only.split(',')
        scenes = {k: v for k, v in scenes.items() if k in keys}
    rows = []
    for name, cfg in scenes.items():
        r = run_scene(name, cfg, policy_budget=a.policy_budget,
                      optimize_budget=a.optimize_budget)
        rows.append(r)
        print(f"{r['scene']:16s} fill={r['fill']:6.2f} (loose={r['fill_loose']:6.2f}) "
              f"norm={r['norm']:5.1f}% packed={r['packed']:3d}/{r['total']:3d} "
              f"({r['pct']:5.1f}%) cogz={r['cog_h']:.2f} topz={r['top_z']:.2f} "
              f"prio_bur={r['n_prio_buried']} soft_bur={r['n_soft_buried']} "
              f"prio_wrong={r['n_prio_wrong']} maxpol={r['max_policy_s']:.2f}s "
              f"wall={r['wall_s']:.0f}s", flush=True)
    if rows:
        print(f"\nMEAN norm% = {sum(r['norm'] for r in rows)/len(rows):.2f}  "
              f"MIN = {min(r['norm'] for r in rows):.2f}  "
              f"MEAN packed% = {sum(r['pct'] for r in rows)/len(rows):.1f}  "
              f"MEAN fill = {sum(r['fill'] for r in rows)/len(rows):.2f}  "
              f"MEAN fill_loose = {sum(r['fill_loose'] for r in rows)/len(rows):.2f}")
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
