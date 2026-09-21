"""Cross-entropy method over the agent's scoring weights.

Why CEM rather than Q-learning, for this problem:

* The reward is a step function with wide flat plateaus (every
  threshold-safe configuration measured so far lands in a 3.4-point band),
  which is the worst possible gradient signal and the best possible case
  for a direct, derivative-free search.
* One episode costs 10-30s and there is no faster surrogate, so the budget
  is ~10^2 evaluations, not the 10^4-10^5 a value-learner needs.
* The policy is already a linear model in a handful of features, so the
  thing to learn is ~7-14 numbers -- shippable as constants, with no
  PyTorch (the evaluation image has pybullet/gymnasium/pillow and nothing
  else).

Every candidate in a generation is evaluated on the *same* scenes (common
random numbers), so differences between candidates are not confounded by
scene difficulty. The whole (candidate x scene) matrix is submitted as one
job pool rather than a loop of per-candidate pools, which keeps all workers
busy through the tail of each generation.

State is written after every generation, so a run can be resumed with
--resume after an interruption.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/takato/comp/nedo'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import objective as obj  # noqa: E402

# (module, name, initial, scale, lo, hi). 'log' parameters are searched in
# log space because they span orders of magnitude and are strictly positive.
PARAM_SETS = {
    # The six placement/selection weights plus the support threshold. All of
    # them were hand-tuned for fill and survival; none has ever been fitted
    # to the composite-through-the-cliff objective.
    'base7': [
        ('p', 'W_LOW', 400.0, 'log', 50.0, 3000.0),
        ('p', 'W_BACK', 200.0, 'log', 20.0, 2000.0),
        ('p', 'W_FLAT', 600.0, 'log', 50.0, 5000.0),
        ('p', 'W_LEFT', 150.0, 'log', 10.0, 2000.0),
        ('p', 'W_MASS_HIGH', 0.6, 'log', 0.05, 200.0),
        ('p', 'W_ITEM_VOL', 3000.0, 'log', 100.0, 30000.0),
        ('p', 'SUPPORT_MIN_COVER', 0.60, 'lin', 0.30, 0.95),
    ],
}
# The seven new features ship at weight 0, i.e. inert, so 'full14' is a
# strict superset of 'base7' and starts from exactly the base7 policy.
PARAM_SETS['full14'] = PARAM_SETS['base7'] + [
    ('p', 'W_WASTE', 0.0, 'lin', -300.0, 3000.0),
    ('p', 'W_SUPPORT', 0.0, 'lin', -300.0, 3000.0),
    ('p', 'W_WALL', 0.0, 'lin', -300.0, 3000.0),
    ('p', 'W_SEAM', 0.0, 'lin', -1500.0, 1500.0),
    ('p', 'W_CEIL', 0.0, 'lin', -1500.0, 1500.0),
    ('p', 'W_SOFT_DEFER', 0.0, 'lin', -3000.0, 3000.0),
    ('p', 'W_PRIO_DEFER', 0.0, 'lin', -3000.0, 3000.0),
]
# The full14 CEM run (2026-09-05/06, cem_full14.json) is what's shipped
# today -- so the *next* run should continue from that best individual, not
# restart from the pre-CEM hand-tuned point above. 'full15' does that, and
# adds one new candidate feature at weight 0: W_SIDE (side-adjacent item
# contact), the strongest single lever the 2026-09-13 mechanism study found
# (docs/2026-09-13-survey戦略.md P2, tools/mechanism_log.py) -- isolated
# items moved 0.165m/78% under shake, items with side contact 0.104m/56%,
# far more than height in the stack or container-wall contact moved either.
# Per this project's own rule (packer.py's block comment above W_WASTE):
# add candidates as inert features and let joint optimisation judge them,
# never a one-at-a-time A/B.
PARAM_SETS['full15'] = [
    ('p', 'W_LOW', 415.8, 'log', 50.0, 3000.0),
    ('p', 'W_BACK', 326.87, 'log', 20.0, 2000.0),
    ('p', 'W_FLAT', 1368.9, 'log', 50.0, 5000.0),
    ('p', 'W_LEFT', 130.16, 'log', 10.0, 2000.0),
    ('p', 'W_MASS_HIGH', 0.50605, 'log', 0.05, 200.0),
    ('p', 'W_ITEM_VOL', 1351.2, 'log', 100.0, 30000.0),
    ('p', 'SUPPORT_MIN_COVER', 0.62978, 'lin', 0.30, 0.95),
    ('p', 'W_WASTE', 107.04, 'lin', -300.0, 3000.0),
    ('p', 'W_SUPPORT', 222.17, 'lin', -300.0, 3000.0),
    ('p', 'W_WALL', 67.088, 'lin', -300.0, 3000.0),
    ('p', 'W_SEAM', -332.72, 'lin', -1500.0, 1500.0),
    ('p', 'W_CEIL', 469.29, 'lin', -1500.0, 1500.0),
    ('p', 'W_SOFT_DEFER', -269.02, 'lin', -3000.0, 3000.0),
    ('p', 'W_PRIO_DEFER', 481.4, 'lin', -3000.0, 3000.0),
    ('p', 'W_SIDE', 0.0, 'lin', -300.0, 3000.0),
]
# Re-fit the shipped full14 vector for a world where the space under the
# container shelf is reachable (docs/2026-09-19-実装の総括.md, and the
# diagnosis in memory: opening one jammed shelf scene showed 0.91 m3 --
# 22.5% of the container -- holding zero boxes, because USE_UNDER_OVERHANG
# is off and so _landing_under never proposes a candidate there).
#
# That flag was measured and turned off for a documented reason: burying
# items under the shelf buries soft and priority ones, costing ~10 points
# of soft_item_score, and real task 001 went 64.3% -> 54.8% packed. But
# that was a *single toggle* against weights fitted for the flag being
# OFF -- and the two weights that govern the named failure mechanism,
# W_SOFT_DEFER and W_PRIO_DEFER, are both in this search. Joint fitting is
# exactly what turned 0-for-24 single-feature A/Bs into full14's +6.34.
#
# USE_UNDER_OVERHANG is pinned rather than searched: lo == hi == 1.0 makes
# its sigma zero and from_search clamp it, so every candidate in every
# generation runs with the under-shelf candidates switched on.
#
# W_SIDE is deliberately NOT carried over from full15. It was measured on
# 2026-09-19 and moves its own mechanism backwards -- raising it made shake
# displacement worse (0.168 -> 0.195 m), not better -- so it would only
# cost a dimension of search.
PARAM_SETS['undershelf'] = [
    ('p', 'W_LOW', 415.8, 'log', 50.0, 3000.0),
    ('p', 'W_BACK', 326.87, 'log', 20.0, 2000.0),
    ('p', 'W_FLAT', 1368.9, 'log', 50.0, 5000.0),
    ('p', 'W_LEFT', 130.16, 'log', 10.0, 2000.0),
    ('p', 'W_MASS_HIGH', 0.50605, 'log', 0.05, 200.0),
    ('p', 'W_ITEM_VOL', 1351.2, 'log', 100.0, 30000.0),
    ('p', 'SUPPORT_MIN_COVER', 0.62978, 'lin', 0.30, 0.95),
    ('p', 'W_WASTE', 107.04, 'lin', -300.0, 3000.0),
    ('p', 'W_SUPPORT', 222.17, 'lin', -300.0, 3000.0),
    ('p', 'W_WALL', 67.088, 'lin', -300.0, 3000.0),
    ('p', 'W_SEAM', -332.72, 'lin', -1500.0, 1500.0),
    ('p', 'W_CEIL', 469.29, 'lin', -1500.0, 1500.0),
    ('p', 'W_SOFT_DEFER', -269.02, 'lin', -3000.0, 3000.0),
    ('p', 'W_PRIO_DEFER', 481.4, 'lin', -3000.0, 3000.0),
    ('p', 'USE_UNDER_OVERHANG', 1.0, 'lin', 1.0, 1.0),
]


def to_search(v, kind):
    return math.log(max(v, 1e-9)) if kind == 'log' else v


def from_search(z, kind, lo, hi):
    v = math.exp(z) if kind == 'log' else z
    return min(max(v, lo), hi)


def _one(args):
    """Run one (candidate, scene) pair in a worker."""
    cand_idx, name, cfg, overrides, pol, opt, agent_dir = args
    os.environ['AGENT_DIR'] = agent_dir
    for p in (HERE, agent_dir, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import geometry, packer  # noqa: F401
    for mod, key, val in overrides:
        setattr({'g': geometry, 'p': packer}[mod], key, val)
    import harness
    row = harness.run_scene(name, cfg, policy_budget=pol, optimize_budget=opt)
    return cand_idx, row


def evaluate(cands, scenes, jobs, pol, opt, agent_dir, spec):
    """Evaluate every candidate on every scene, as one flat job pool."""
    tasks = []
    for i, values in enumerate(cands):
        ov = [(spec[j][0], spec[j][1], values[j]) for j in range(len(spec))]
        for name, cfg in scenes.items():
            tasks.append((i, name, cfg, ov, pol, opt, agent_dir))
    rows = [[] for _ in cands]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for cand_idx, row in ex.map(_one, tasks):
            rows[cand_idx].append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_pool.json',
                    help='comma-separated scene files; they are pooled into one training set')
    ap.add_argument('--limit', type=int, default=0, help='use only the first N scenes')
    ap.add_argument('--params', default='base7', choices=sorted(PARAM_SETS))
    ap.add_argument('--pop', type=int, default=14)
    ap.add_argument('--gens', type=int, default=8)
    ap.add_argument('--elite-frac', type=float, default=0.25)
    ap.add_argument('--sigma', type=float, default=0.45, help='initial sigma (log-space units)')
    ap.add_argument('--sigma-lin', type=float, default=0.15, help='initial sigma for lin params, as a fraction of range')
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--policy-budget', type=float, default=5.0)
    ap.add_argument('--optimize-budget', type=float, default=60.0)
    ap.add_argument('--agent-dir', default=os.path.join(ROOT, 'agents', 'submit'))
    ap.add_argument('--seed', type=int, default=20260903)
    ap.add_argument('--out', default='cem_run.json')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--step-objective', dest='smooth', action='store_false',
                    help='optimise the hard cliff (overfits; see objective.py)')
    ap.set_defaults(smooth=True)
    a = ap.parse_args()

    spec = PARAM_SETS[a.params]
    scenes = {}
    for f in a.scenes.split(','):
        scenes.update(json.load(open(f.strip())))
    if a.limit:
        scenes = {k: scenes[k] for k in sorted(scenes)[:a.limit]}
    score_fn = obj.smooth_objective if a.smooth else obj.objective
    print(f'training on {len(scenes)} scenes from {a.scenes}, '
          f'objective={"smooth" if a.smooth else "step"}', flush=True)

    rng = random.Random(a.seed)
    mean = [to_search(s[2], s[3]) for s in spec]
    sigma = []
    for s in spec:
        sigma.append(a.sigma if s[3] == 'log' else a.sigma_lin * (s[5] - s[4]))

    state = {'params': a.params, 'spec': [list(s) for s in spec], 'scenes': sorted(scenes),
             'gens': [], 'best': None}
    if a.resume and os.path.exists(a.out):
        state = json.load(open(a.out))
        mean, sigma = state['mean'], state['sigma']
        print(f"resumed at generation {len(state['gens'])}", flush=True)

    t_start = time.time()
    for gen in range(len(state['gens']), a.gens):
        cands = []
        if gen == 0 and not state['gens']:
            cands.append([from_search(m, s[3], s[4], s[5]) for m, s in zip(mean, spec)])
        while len(cands) < a.pop:
            z = [m + sg * rng.gauss(0, 1) for m, sg in zip(mean, sigma)]
            cands.append([from_search(zi, s[3], s[4], s[5]) for zi, s in zip(z, spec)])

        t0 = time.time()
        rows = evaluate(cands, scenes, a.jobs, a.policy_budget, a.optimize_budget,
                        a.agent_dir, spec)
        scored = sorted(((score_fn(r), i) for i, r in enumerate(rows)), reverse=True)
        n_elite = max(2, int(round(a.elite_frac * len(cands))))
        elite = [cands[i] for _, i in scored[:n_elite]]

        # Refit in search space, with a variance floor so the search cannot
        # collapse onto one point while the objective is still this flat.
        new_mean, new_sigma = [], []
        for j, s in enumerate(spec):
            zs = [to_search(e[j], s[3]) for e in elite]
            mu = sum(zs) / len(zs)
            var = sum((z - mu) ** 2 for z in zs) / max(len(zs) - 1, 1)
            floor = 0.10 if s[3] == 'log' else 0.03 * (s[5] - s[4])
            new_mean.append(mu)
            new_sigma.append(max(math.sqrt(var), floor))
        mean, sigma = new_mean, new_sigma

        best_score, best_i = scored[0]
        rec = {
            'gen': gen, 'seconds': round(time.time() - t0, 1),
            'best_score': best_score,
            'best_params': {s[1]: cands[best_i][j] for j, s in enumerate(spec)},
            'best_summary': obj.summary(rows[best_i]),
            'scores': [sc for sc, _ in scored],
            'mean': {s[1]: from_search(m, s[3], s[4], s[5]) for m, s in zip(mean, spec)},
        }
        state['gens'].append(rec)
        state['mean'], state['sigma'] = mean, sigma
        if state['best'] is None or best_score > state['best']['best_score']:
            state['best'] = rec
        json.dump(state, open(a.out, 'w'), indent=1)

        s = rec['best_summary']
        print(f"gen {gen}: best={best_score:.3f}  packed={s['packed']:.1f}%  "
              f"comp={s['composite']:.2f}  below46={s['below46']}  "
              f"({rec['seconds']:.0f}s, total {(time.time()-t_start)/3600:.2f}h)", flush=True)
        print('   ' + '  '.join(f'{k}={v:.4g}' for k, v in rec['best_params'].items()), flush=True)

    print('\nBEST OVERALL', json.dumps(state['best']['best_params'], indent=1), flush=True)
    print('  summary', json.dumps(state['best']['best_summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
