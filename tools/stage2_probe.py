"""Does a physics second stage rank orders better than the rollout does?

Two judges have been through here. The first settled the *finished* packing
in one drop and scored that; it ranked worse than no judge at all (rho 0.234
against the cheap rollout's 0.337) because a whole stack landing at once is
not the physics of one item at a time. The second -- tools/physrollout.py --
places one item at a time and re-derives each placement against what physics
left behind, which is what the real episode does. Select with --judge.

order_fidelity.py measured the ceiling for two-stage selection: letting an
oracle pick from the rollout's top 3 is worth +2.12 composite over what
optimize() returns today, which is as large as any change adopted so far.
That ceiling is only reachable with a judge that ranks orders better than
the rollout's Spearman +0.49.

The rollout cannot be that judge -- it is deterministic, so re-running it
says the same thing, and it has no way to see stability at all. Physics can.
This drops the rollout's *planned* packing into a PyBullet world built the
same way agents/submit/twin.py builds one, settles it, shakes it with the
same lateral-gravity protocol tools/scores.py uses, and scores all five
metrics off the settled poses.

For each scene it then reports, within the scene:

    rho(rollout score, true episode score)     the stage-1 ranking, ~0.49
    rho(physics score, true episode score)     the candidate stage-2 ranking

plus what each would actually have returned. A stage 2 is only worth
building into the agent if the second number is clearly the larger.

    ../.venv/bin/python stage2_probe.py --n 12 --orders 8 --jobs 12
"""
import argparse, contextlib, io, json, math, os, random, shutil, statistics, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']

PATCH = ("""            self._plan = best_plan""",
         """            self._rollout = rollout
            self._specs_of = specs_of
            self._plan = best_plan""")


def build(dst):
    shutil.copytree(os.path.join(ROOT, 'agents', 'submit'), dst,
                    ignore=shutil.ignore_patterns('__pycache__'))
    p = os.path.join(dst, 'agent.py')
    s = open(p).read()
    if PATCH[0] not in s:
        raise SystemExit('stage2_probe: agent.py drifted, anchor missing')
    open(p, 'w').write(s.replace(*PATCH, 1))
    return dst


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def physics_score(container_dicts, geoms, boxes_by_c, packer):
    """Settle a planned packing, shake it, and score the five metrics.

    boxes_by_c: {container_index: [(spec, (cx, cy, cz) local, orn_idx), ...]}
    """
    import pybullet as p
    import twin as twin_mod
    total = {k: 0.0 for k in METRICS}
    denom = 0.0
    counted_vol = 0.0
    mass = moment = height = 0.0
    disp_all = []
    settled = []          # (cdict, spec, world pos, half extents)
    for cd in container_dicts:
        cidx = cd['index']
        plan = boxes_by_c.get(cidx, [])
        if not plan:
            continue
        g = geoms[cidx]
        t = twin_mod.Twin(p, cd, g['static_boxes'])
        try:
            ids = []
            for spec, pos, orn in plan:
                quat = p.getQuaternionFromEuler(twin_mod.ORNS[orn])
                col = t.cl.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[spec['length'] / 2.0,
                                             spec['width'] / 2.0,
                                             spec['height'] / 2.0])
                bid = t.cl.createMultiBody(spec.get('mass', 1.0), col,
                                           basePosition=list(pos), baseOrientation=quat)
                t.cl.changeDynamics(bid, -1, **twin_mod._dyn(spec))
                ids.append((bid, spec))
            for _ in range(240):
                t.cl.stepSimulation()
            before = {}
            for bid, spec in ids:
                bp, bo = t.cl.getBasePositionAndOrientation(bid)
                before[bid] = bp
                R = t.cl.getMatrixFromQuaternion(bo)
                hx = (abs(R[0]) * spec['length'] + abs(R[1]) * spec['width']
                      + abs(R[2]) * spec['height']) / 2.0
                hy = (abs(R[3]) * spec['length'] + abs(R[4]) * spec['width']
                      + abs(R[5]) * spec['height']) / 2.0
                hz = (abs(R[6]) * spec['length'] + abs(R[7]) * spec['width']
                      + abs(R[8]) * spec['height']) / 2.0
                settled.append((cidx, g, spec, bp, (hx, hy, hz)))
            # shake, mirroring tools/scores.py: lateral 0.3g, 4 directions x 2
            a = 9.8 * 0.30
            for _ in range(2):
                for gx, gy in ((a, 0), (-a, 0), (0, a), (0, -a)):
                    t.cl.setGravity(gx, gy, -9.8)
                    for _ in range(150):
                        t.cl.stepSimulation()
            t.cl.setGravity(0, 0, -9.8)
            for _ in range(200):
                t.cl.stepSimulation()
            for bid, spec in ids:
                bp, _ = t.cl.getBasePositionAndOrientation(bid)
                disp_all.append(math.dist(bp, before[bid]))
        finally:
            t.close()
        denom += packer.usable_volume(g)
        height = max(height, g['height'])
    if not settled:
        return None
    boxes = []
    for cidx, g, spec, bp, half in settled:
        v = 8.0 * half[0] * half[1] * half[2]
        on_floor = abs((bp[2] - half[2]) - g['floor_struct_z']) < 0.03
        if not on_floor:
            counted_vol += v
        m = spec.get('mass', 1.0)
        mass += m
        moment += m * bp[2]
        boxes.append((cidx, {'center': bp, 'half': half, 'mass': m,
                             'is_soft': spec.get('is_soft', False),
                             'is_prioritized': spec.get('is_prioritized', False)}))
    any_prio = any(geoms[c]['is_prioritized'] for c, _ in boxes)

    def cls(attr, check):
        members = [(c, b) for c, b in boxes if b[attr]]
        if not members:
            return 100.0
        bad = 0
        for c, b in members:
            top = b['center'][2] + b['half'][2]
            for oc, o in boxes:
                if o is b or oc != c or o[attr] == b[attr]:
                    continue
                if o['center'][2] - o['half'][2] < top - 0.02:
                    continue
                if (abs(o['center'][0] - b['center'][0]) < o['half'][0] + b['half'][0] and
                        abs(o['center'][1] - b['center'][1]) < o['half'][1] + b['half'][1]):
                    bad += 1
                    break
        wrong = sum(1 for c, b in members if not geoms[c]['is_prioritized']) if check else 0
        return 100.0 * (1.0 - (bad + wrong) / (2.0 * len(members)))

    mean_d = statistics.mean(disp_all) if disp_all else 0.0
    total['fill'] = min(100.0 * counted_vol / denom, 100.0) if denom else 0.0
    total['cog_score'] = max(0.0, min(100.0, 100.0 * (1.0 - (moment / mass) / height)))
    total['stability_score'] = 100.0 * math.exp(-mean_d / 0.10)
    total['placement_score'] = cls('is_prioritized', any_prio)
    total['soft_item_score'] = cls('is_soft', False)
    return sum(total[k] for k in METRICS) / 5.0, total


JUDGE = 'seq'
SETTLE = 80


def _one(job):
    name, cfg, agent_dir, n_orders, seed, judge, settle = job
    global JUDGE, SETTLE
    JUDGE, SETTLE = judge, settle
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, agent_dir)
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    import packer
    import scores
    A.OPTIMIZE_TIME_BUDGET = 8.0
    rng = random.Random(seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = A.Agent(module_path=agent_dir)
        env.reset_settings()
        init = env.get_init_states()
        ag.get_init_states(init)
        best = ag.optimize(env.get_info_for_optimization())
        container_dicts = init['container_list']
        geoms = {c['index']: packer.container_geometry(c) for c in container_dicts}
        env.close()
        rollout, specs_of = ag._rollout, ag._specs_of
        n = len(best)
        orders = [list(best)]
        while len(orders) < n_orders:
            o = list(best)
            for _ in range(rng.randint(1, 3)):
                k = rng.randint(2, max(2, n // 6))
                i = rng.randrange(max(1, n - k))
                seg = o[i:i + k]
                del o[i:i + k]
                o[rng.randrange(len(o) + 1):0] = seg
            if len(o) == n and set(o) == set(best):
                orders.append(o)

        pred, phys, real = [], [], []
        for o in orders:
            specs = specs_of(o)
            key, _, plan, _ = rollout(specs, time.perf_counter() + 30.0)
            pred.append(key[1])
            by_c = {}
            spec_of = {s['index']: s for s in specs}
            for idx, (cidx, cx, cy, orn) in plan.items():
                spec = spec_of[idx]
                half = __import__('geometry').half_extents(
                    spec['length'], spec['width'], spec['height'], orn)
                # the rollout's committed z is not in the plan, so recover the
                # design bottom from the plan's own landing
                by_c.setdefault(cidx, []).append((spec, (cx, cy, None), orn, half))
            # resolve z by replaying the commits in plan order
            states = {c['index']: packer.ContainerState(c) for c in container_dicts}
            placed = {}
            for idx in o:
                if idx not in plan:
                    continue
                cidx, cx, cy, orn = plan[idx]
                spec = spec_of[idx]
                half = __import__('geometry').half_extents(
                    spec['length'], spec['width'], spec['height'], orn)
                z = packer.settled_center_z(states[cidx], cx, cy, half)
                states[cidx].commit((cx, cy, z), half, spec)
                placed.setdefault(cidx, []).append((spec, (cx, cy, z), orn))
            if JUDGE == 'seq':
                import physrollout, twin as twin_mod, geometry as geom_mod
                _, pc, _ = physrollout.run(
                    container_dicts, geoms, specs, packer, twin_mod, geom_mod,
                    allowed_misses=max(1, int(cfg['item_stream']['look_ahead'])),
                    settle_steps=SETTLE,
                    disp_threshold=cfg['validator'].get('displacement_threshold', 0.3))
                phys.append(pc)
            else:
                ps = physics_score(container_dicts, geoms, placed, packer)
                phys.append(ps[0] if ps else 0.0)

            e = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
            e.reset_settings()
            ag2 = A.Agent(module_path=agent_dir)
            ag2.get_init_states(e.get_init_states())
            e.set_item_order(o)
            e.reset_item_stream()
            obs, _ = e.reset(seed=42)
            term, steps = False, 0
            while not term and steps < 500:
                obs, _, term, _, _ = e.step(ag2.policy(observation=obs))
                steps += 1
            rep = e.evaluate()
            truth = scores.all_scores(e)
            row = dict(fill=rep['fill_score'], **{k: truth[k] for k in METRICS[1:]})
            real.append(sum(row[k] for k in METRICS) / 5.0)
            e.close()
    return dict(scene=name, pred=pred, phys=phys, real=real,
                rho_pred=spearman(pred, real), rho_phys=spearman(phys, real))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--orders', type=int, default=8)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--settle', type=int, default=80,
                    help='settle steps per placement; the real config ships 300')
    ap.add_argument('--judge', default='seq', choices=['seq', 'drop'],
                    help="'seq' = sequential physics rollout (physrollout.py); "
                         "'drop' = settle the finished packing in one go")
    a = ap.parse_args()
    scenes = json.load(open(os.path.join(ROOT, 'tools', a.scenes)))
    scenes = {k: v for k, v in scenes.items() if v['agent'].get('optimize')}
    names = list(scenes)[:a.n]
    tmp = tempfile.mkdtemp(prefix='stage2_')
    try:
        agent_dir = build(os.path.join(tmp, 'submit'))
        jobs = [(nm, scenes[nm], agent_dir, a.orders, 7000 + i, a.judge, a.settle)
                for i, nm in enumerate(names)]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            rows = list(ex.map(_one, jobs))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(rows, open(os.path.join(ROOT, 'tools', 'stage2_probe.json'), 'w'), indent=1)

    print(f"{len(rows)} scenes x {a.orders} orders, judge={a.judge}, "
          f"settle={a.settle}\n")
    print(f"{'scene':8} {'rho rollout':>12} {'rho physics':>12}")
    for r in rows:
        print(f"{r['scene']:8} {r['rho_pred']:12.2f} {r['rho_phys']:12.2f}")
    print(f"\nmean rho: rollout {statistics.mean(r['rho_pred'] for r in rows):+.3f}   "
          f"physics {statistics.mean(r['rho_phys'] for r in rows):+.3f}")

    def pick(r, scores_, k=None):
        idx = sorted(range(len(scores_)), key=lambda i: -scores_[i])
        return r['real'][idx[0]]
    cur = statistics.mean(pick(r, r['pred']) for r in rows)
    st2 = statistics.mean(
        max(r['real'][i] for i in sorted(range(len(r['pred'])),
                                         key=lambda i: -r['pred'][i])[:3]
            if True) for r in rows) if False else None
    # stage 2 over the rollout's top 3
    vals = []
    for r in rows:
        top3 = sorted(range(len(r['pred'])), key=lambda i: -r['pred'][i])[:3]
        best_by_phys = max(top3, key=lambda i: r['phys'][i])
        vals.append(r['real'][best_by_phys])
    oracle = statistics.mean(
        max(r['real'][i] for i in sorted(range(len(r['pred'])),
                                         key=lambda i: -r['pred'][i])[:3]) for r in rows)
    rand = statistics.mean(statistics.mean(r['real']) for r in rows)
    print(f"\nmean true composite of the order returned:")
    print(f"  random order                      {rand:6.2f}")
    print(f"  rollout argmax (today)            {cur:6.2f}")
    print(f"  physics judge over rollout top 3  {statistics.mean(vals):6.2f}  "
          f"({statistics.mean(vals) - cur:+.2f})")
    print(f"  oracle over rollout top 3         {oracle:6.2f}  ({oracle - cur:+.2f})")


if __name__ == '__main__':
    main()
