"""Verify a CEM run on data it never saw.

The base7 run is the reason this exists. It gained +3.50 on its training
pool and lost 0.87 on four unseen ones: the best of 112 noisy evaluations
is biased upward, and the step objective let it memorise which training
scenes sat on the item-count cliff. So a CEM result means nothing until it
has been re-measured on pools that were not in --scenes, and on the two
real tasks.

Reports both the best individual and the final distribution mean. CEM's
actual output is the mean; the best sample is the one carrying the
winner's curse, and the gap between them is a direct read on how much of
the training gain was luck.
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import objective as obj  # noqa: E402

VENV = '/home/takato/comp/nedo/.venv/bin/python'


def sets_for(params):
    # Two argv elements per override: subprocess takes a list, not a shell
    # string, so '--set p:K=V' as one element is passed as a single unknown
    # option and argparse exits 2.
    out = []
    for k, v in params.items():
        out += ['--set', f'p:{k}={v:.10g}']
    return out


def run(scenes, tag, params, jobs, opt_budget):
    out = os.path.join(HERE, f'ab_{tag}.json')
    if not os.path.exists(out):
        cmd = [VENV, 'ab.py', '--scenes', scenes, '--tag', tag, '--jobs', str(jobs),
               '--optimize-budget', str(opt_budget)] + sets_for(params)
        subprocess.run(cmd, cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='cem_full14.json')
    ap.add_argument('--train', default='scenes_pool.json,scenes_shelf.json')
    ap.add_argument('--holdout', default='scenes_k.json,scenes_ktest.json,scenes_shelftest.json')
    ap.add_argument('--baselines', default='scenes_k.json=W_k_pre,scenes_ktest.json=W_ktest_pre,'
                                           'scenes_shelftest.json=W_shelftest_pre')
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--optimize-budget', type=float, default=60.0)
    ap.add_argument('--prefix', default='V14')
    a = ap.parse_args()

    st = json.load(open(a.state))
    cand = {'best': st['best']['best_params'],
            'mean': st['gens'][-1]['mean']}
    base_of = dict(kv.split('=') for kv in a.baselines.split(','))

    print(f"state={a.state}  generations={len(st['gens'])}")
    for k, v in cand['best'].items():
        print(f"  best {k:<20} {v:.5g}   mean {cand['mean'][k]:.5g}")

    print(f"\n{'pool':<22}{'baseline':>10}{'best':>10}{'mean':>10}   "
          f"{'Δbest':>8}{'Δmean':>8}   (smooth objective)")
    agg = {'best': [], 'mean': []}
    for scenes in a.holdout.split(','):
        scenes = scenes.strip()
        b = json.load(open(os.path.join(HERE, f'ab_{base_of[scenes]}.json')))
        base = obj.smooth_objective(b)
        line = f'{scenes.replace("scenes_", "").replace(".json", ""):<22}{base:10.2f}'
        for which in ('best', 'mean'):
            rows = run(scenes, f'{a.prefix}_{which}_{os.path.basename(scenes)[7:-5]}',
                       cand[which], a.jobs, a.optimize_budget)
            v = obj.smooth_objective(rows)
            agg[which].append(v - base)
            line += f'{v:10.2f}'
        line += f'   {agg["best"][-1]:+8.2f}{agg["mean"][-1]:+8.2f}'
        print(line)

    print()
    for which in ('best', 'mean'):
        d = agg[which]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1))
        print(f'  未使用プール平均 Δsmooth_objective ({which}) = {m:+.2f}  '
              f'(プール間 sd {sd:.2f}, n={len(d)})')

    print('\n実タスク（決定論的だが R001 は不安定シーン。単独で採否を決めないこと）')
    for which in ('best', 'mean'):
        rows = run('scenes_real.json', f'{a.prefix}_{which}_real', cand[which], 2, 150.0)
        for r in sorted(rows, key=lambda z: z['scene']):
            print(f"  {which:<5} {r['scene']}  comp={obj.composite(r):5.2f}  "
                  f"packed={r['pct']:5.1f}%  fill={r['fill']:5.2f}")


if __name__ == '__main__':
    main()
