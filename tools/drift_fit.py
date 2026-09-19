"""Measure the gap between where the agent asks for an item and where it ends up.

optimize()'s rollout commits every item at its design position and never
moves it again. The real episode drops each item from start_z above the
target and lets physics settle it, so the state the next decision sees is
not the state the rollout assumed. order_fidelity.py measured the
consequence -- within one scene the rollout ranks arrival orders at Spearman
+0.27 on composite and +0.47 on items placed -- and this measures the cause.

For every accepted placement it records the requested place_pos and the
item's settled pose, so a drift model can be fitted from data instead of
guessed.

    ../.venv/bin/python drift_fit.py --scenes scenes_off.json --n 16 --jobs 12
"""
import argparse, contextlib, io, json, os, statistics, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _one(job):
    name, cfg, budget = job
    agent_dir = os.path.join(ROOT, 'agents', 'submit')
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, agent_dir)
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    A.OPTIMIZE_TIME_BUDGET = budget
    out = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = A.Agent(module_path=agent_dir)
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if cfg['agent'].get('optimize'):
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, _ = env.reset(seed=42)
        term, steps = False, 0
        while not term and steps < 500:
            action = ag.policy(observation=obs)
            pool = obs.get('pool_list') or []
            idx = int(action.get('item_idx', -1))
            want = list(map(float, action.get('place_pos', [0, 0, 0])))
            item_index = pool[idx]['index'] if 0 <= idx < len(pool) else None
            obs, _, term, _, _ = env.step(action)
            steps += 1
            if item_index is None:
                continue
            for c in env.container_manager.containers:
                for it in c.packed_items:
                    if it.index == item_index and it.pos is not None:
                        out.append(dict(scene=name, item=item_index,
                                        # place_pos is container-local in x;
                                        # the env adds the container centre.
                                        dx=it.pos[0] - want[0] - c.center[0],
                                        dy=it.pos[1] - want[1],
                                        dz=it.pos[2] - want[2],
                                        h=it.height, mass=it.mass,
                                        soft=bool(it.is_soft),
                                        n_before=len(c.packed_items) - 1))
                        break
        env.close()
    return out


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--budget', type=float, default=20.0)
    args = ap.parse_args()

    scenes = json.load(open(os.path.join(ROOT, 'tools', args.scenes)))
    scenes = {k: v for k, v in scenes.items() if v['agent'].get('optimize')}
    names = list(scenes)[:args.n]
    jobs = [(n, scenes[n], args.budget) for n in names]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        rows = [r for part in ex.map(_one, jobs) for r in part]
    json.dump(rows, open(os.path.join(ROOT, 'tools', 'drift_fit.json'), 'w'))

    print(f"{len(rows)} accepted placements over {len(names)} scenes\n")
    print(f"{'axis':6} {'mean':>9} {'sd':>8} {'|d| p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    for ax in ('dx', 'dy', 'dz'):
        v = [r[ax] for r in rows]
        a = [abs(x) for x in v]
        print(f"{ax:6} {statistics.mean(v):+9.4f} {statistics.pstdev(v):8.4f} "
              f"{q(a, 0.5):9.4f} {q(a, 0.9):9.4f} {q(a, 0.99):9.4f} {max(a):9.4f}")
    big = sum(1 for r in rows if max(abs(r['dx']), abs(r['dy']), abs(r['dz'])) > 0.01)
    print(f"\nplacements that moved more than 1cm in some axis: {big}/{len(rows)} "
          f"({100.0 * big / max(len(rows), 1):.0f}%)")
    print(f"\nsplit by item class:")
    for label, sel in (('soft', lambda r: r['soft']), ('rigid', lambda r: not r['soft'])):
        v = [r['dz'] for r in rows if sel(r)]
        if not v:
            continue
        a = [abs(x) for x in v]
        print(f"  {label:6} n={len(v):4d}  dz mean {statistics.mean(v):+.4f}  "
              f"sd {statistics.pstdev(v):.4f}  |dz| p50 {q(a, 0.5):.4f}  p90 {q(a, 0.9):.4f}  "
              f"p99 {q(a, 0.99):.4f}")
    deep = [r for r in rows if r['dz'] < -0.01]
    if deep:
        print(f"\nsinks deeper than 1cm: {len(deep)}/{len(rows)} "
              f"({100.0 * len(deep) / len(rows):.0f}%), of which "
              f"{100.0 * sum(1 for r in deep if r['soft']) / len(deep):.0f}% are soft "
              f"(soft share overall {100.0 * sum(1 for r in rows if r['soft']) / len(rows):.0f}%)")


if __name__ == '__main__':
    main()
