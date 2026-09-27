"""Calibrate the work clock (packer.CLOCK_MODE='work') against wall time.

Runs scenes with an agent in 'wall' mode, records for every best_placement
call the work counters it added (packer.WORK_STATS) and the wall time it
took, then fits non-negative weights

    seconds ~ a*corners + b*corner_boxes + c*valid + d*valid_boxes + e*waste

and reports how well per-policy-call work predicts per-policy-call time
(coefficient of variation of time / predicted). Run it at the parallel load
the search constants were tuned under (~15 workers), since that is the
speed the work budget has to reproduce.

    AGENT_DIR=../agents/det python clock_calib.py --scenes a.json,b.json --jobs 15 --out calib.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = '/home/takato/comp/nedo'
FIELDS = ('corners', 'corner_boxes', 'valid', 'valid_boxes', 'waste')


def _one(args):
    name, cfg, agent_dir, pol = args
    sys.path.insert(0, ROOT)
    sys.path.insert(0, agent_dir)
    import io
    import contextlib
    import importlib
    import packer
    import agent as agent_module
    importlib.reload(agent_module)
    from src.ground_handling.env import GroundHandlingEnv
    packer.CLOCK_MODE = 'wall'
    agent_module.POLICY_TIME_BUDGET = pol
    calls, policies = [], []
    orig_bp = packer.best_placement

    def wrapped(*a, **k):
        s0 = dict(packer.WORK_STATS)
        t0 = time.perf_counter()
        r = orig_bp(*a, **k)
        dt = time.perf_counter() - t0
        calls.append([packer.WORK_STATS[f] - s0[f] for f in FIELDS] + [dt])
        return r
    packer.best_placement = wrapped
    agent_module.best_placement = wrapped
    with contextlib.redirect_stdout(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = agent_module.Agent(module_path='')
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            agent_module.OPTIMIZE_TIME_BUDGET = 20.0
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        term = False
        steps = 0
        while not term and steps < 300:
            s0 = dict(packer.WORK_STATS)
            t0 = time.perf_counter()
            a = ag.policy(observation=obs)
            dt = time.perf_counter() - t0
            policies.append([packer.WORK_STATS[f] - s0[f] for f in FIELDS] + [dt])
            obs, r, term, trunc, info = env.step(a)
            steps += 1
        env.close()
    return dict(scene=name, calls=calls, policies=policies)


def nnls(X, y, iters=5000):
    """Projected-gradient non-negative least squares (no scipy on the box)."""
    scale = X.max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    w = np.maximum(np.linalg.lstsq(Xs, y, rcond=None)[0], 0.0)
    L = np.linalg.norm(Xs, 2) ** 2
    for _ in range(iters):
        g = Xs.T @ (Xs @ w - y)
        w = np.maximum(w - g / L, 0.0)
    return w / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--only', default=None)
    ap.add_argument('--jobs', type=int, default=15)
    ap.add_argument('--policy-budget', type=float, default=5.0)
    ap.add_argument('--out', default='clock_calib.json')
    a = ap.parse_args()
    agent_dir = os.environ.get('AGENT_DIR') or os.path.join(ROOT, 'agents', 'det')
    scenes = {}
    for p in a.scenes.split(','):
        scenes.update(json.load(open(p)))
    if a.only:
        keep = a.only.split(',')
        scenes = {k: v for k, v in scenes.items() if any(k.startswith(x) for x in keep)}
    jobs = [(n, c, agent_dir, a.policy_budget) for n, c in scenes.items()]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_one, jobs))
    json.dump(rows, open(a.out, 'w'))

    C = np.array([c for r in rows for c in r['calls'] if c[-1] > 1e-4])
    X, y = C[:, :-1], C[:, -1]
    w = nnls(X, y)
    print(f'{len(C)} best_placement calls from {len(rows)} scenes')
    for f, v in zip(FIELDS, w):
        print(f'  {f:13s} {v * 1e6:10.4f} us')
    pred = X @ w
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'  per-call R^2 = {r2:.3f}')
    P = np.array([p for r in rows for p in r['policies'] if p[-1] > 0.05])
    ratio = P[:, -1] / np.maximum(P[:, :-1] @ w, 1e-9)
    print(f'policy calls: {len(P)}  time/predicted median {np.median(ratio):.3f}  '
          f'CV {ratio.std() / ratio.mean():.3f}  p10 {np.percentile(ratio, 10):.3f}  '
          f'p90 {np.percentile(ratio, 90):.3f}')


if __name__ == '__main__':
    main()
