"""Run agent directories over the stress scenes and flag early deaths.

Unlike ab.py a crash in one scene is recorded, not fatal: a crash is
exactly what this is looking for. Timing uses the competition budgets
(policy 5.0 s inside an 8 s limit, optimize 150 s inside 180 s) so that
overruns show up.

    python stress.py --scenes scenes_stress.json --agent-dir DIR --tag T [--jobs 12]
    python stress.py --compare T1 T2          # side by side from stress_<tag>.json
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/takato/comp/nedo'
HARD_LIMIT_S = 8.0      # policy_timeout on the scoring platform (README:151)
WARN_POLICY_S = 6.5


def _one(args):
    name, cfg, agent_dir, pol, opt = args
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, ROOT)
    sys.path.insert(0, agent_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t0 = time.perf_counter()
    try:
        import harness
        r = harness.run_scene(name, cfg, policy_budget=pol, optimize_budget=opt)
        r['error'] = None
    except Exception:
        r = dict(scene=name, error=traceback.format_exc()[-1500:])
    r['wall_s'] = time.perf_counter() - t0
    cl = cfg['containers']['container_list']
    r['n_containers'] = len(cl)
    r['k'] = cfg['item_stream']['look_ahead']
    r['optimize'] = cfg['agent']['optimize']
    return r


def flags(r):
    f = []
    if r.get('error'):
        f.append('EXC')
        return f
    if r.get('max_policy_s', 0) > HARD_LIMIT_S:
        f.append('TIMEOUT')
    elif r.get('max_policy_s', 0) > WARN_POLICY_S:
        f.append('SLOW')
    if r.get('pct', 100) < 40:
        f.append('LOW<40%')
    return f


def run(a):
    scenes = json.load(open(a.scenes))
    if a.only:
        keep = a.only.split(',')
        scenes = {k: v for k, v in scenes.items() if any(k.startswith(p) for p in keep)}
    jobs = [(n, c, a.agent_dir, a.policy_budget, a.optimize_budget) for n, c in scenes.items()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_one, jobs))
    rows.sort(key=lambda r: r['scene'])
    json.dump(rows, open(f'stress_{a.tag}.json', 'w'), indent=1)
    print(f'### {a.tag} agent={a.agent_dir}  [{time.time() - t0:.0f}s]')
    for r in rows:
        if r.get('error'):
            print(f"  {r['scene']:26s} EXC {r['error'].splitlines()[-1][:100]}")
            continue
        print(f"  {r['scene']:26s} packed={r['packed']:3d}/{r['total']:3d} ({r['pct']:5.1f}%) "
              f"fill={r['fill']:5.1f} maxpol={r['max_policy_s']:5.2f}s "
              f"{' '.join(flags(r))}")


def compare(t1, t2):
    A = {r['scene']: r for r in json.load(open(f'stress_{t1}.json'))}
    B = {r['scene']: r for r in json.load(open(f'stress_{t2}.json'))}
    print(f'{"scene":26s} {t1:>18s} {t2:>18s}   dpct')
    for s in sorted(A):
        if s not in B:
            continue
        a, b = A[s], B[s]

        def cell(r):
            if r.get('error'):
                return 'EXC'.rjust(18)
            return f"{r['packed']:3d}/{r['total']:3d} {r['max_policy_s']:4.1f}s".rjust(18)
        d = (b.get('pct', 0) - a.get('pct', 0)) if not (a.get('error') or b.get('error')) else float('nan')
        print(f"{s:26s} {cell(a)} {cell(b)}  {d:+6.1f}  {' '.join(flags(a))} | {' '.join(flags(b))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_stress.json')
    ap.add_argument('--agent-dir', default=os.path.join(ROOT, 'agents', 'submit'))
    ap.add_argument('--tag', default='stress')
    ap.add_argument('--only', default=None, help='comma-separated scene-name prefixes')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--policy-budget', type=float, default=None)
    ap.add_argument('--optimize-budget', type=float, default=None)
    ap.add_argument('--compare', nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)
