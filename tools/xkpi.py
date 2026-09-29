"""Crossing-aware paired comparison of two result sets (A = base, B = treatment).

The leaderboard zeroes every score except fill in a scene that does not
place "a certain number" of items (README:386), so what a change is worth is
how many scenes it pushes over that line, not how much it moves an average.
The line is not published, so it is integrated over a prior:

    F(c) = 0.5 * clip((pct - 50) / 15, 0, 1) + 0.5 * clip((count - 25) / 8, 0, 1)

i.e. an equal mixture of "pct >= T, T ~ U[50, 65]" and "count >= T,
T ~ U[25, 33]". F(c) is the probability the scene clears the line.

PRE-REGISTERED on 2026-09-26 (docs plan v3, B-4). Do not re-fit this prior
to leaderboard results: that would be estimating the threshold, which the
rules forbid.

Main statistic: S = sum_i [F(c_B,i) - F(c_A,i)] (expected extra crossings),
with a paired sign-flip permutation p-value (one-sided, B > A). Per segment
(containers x shelf) the check is "no significant worsening" (one-sided
p < 0.10 for B < A flags it). Secondary: mean deltas of count, pct, fill,
cog, placement, soft with standard errors.

    python xkpi.py ab_BASE.json ab_TREAT.json [--scenes scenes_x.json,...]
    python xkpi.py stress_base.json stress_fix1.json
"""
import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict


def F(pct, count):
    a = min(max((pct - 50.0) / 15.0, 0.0), 1.0)
    b = min(max((count - 25.0) / 8.0, 0.0), 1.0)
    return 0.5 * a + 0.5 * b


def perm_p(diffs, n_perm=20000, seed=0):
    """One-sided p for sum(diffs) > 0 under random sign flips."""
    obs = sum(diffs)
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in nz)
        if s >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _meta(row, scenes):
    cfg = scenes.get(row['scene']) if scenes else None
    if cfg is not None:
        cl = cfg['containers']['container_list']
        return len(cl), any(c.get('require_shelf') for c in cl)
    return row.get('n_containers', '?'), row.get('shelf', '?')


def load(path):
    rows = json.load(open(path))
    return {r['scene']: r for r in rows if not r.get('error')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--scenes', default=None, help='comma-separated scene files for segment metadata')
    ap.add_argument('--perm', type=int, default=20000)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    scenes = {}
    if args.scenes:
        for p in args.scenes.split(','):
            scenes.update(json.load(open(p)))
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit('no common scenes')

    d = [F(B[k]['pct'], B[k]['packed']) - F(A[k]['pct'], A[k]['packed']) for k in keys]
    p = perm_p(d, args.perm)
    ea = sum(F(A[k]['pct'], A[k]['packed']) for k in keys)
    eb = sum(F(B[k]['pct'], B[k]['packed']) for k in keys)
    print(f'n={len(keys)}  E[crossings] A={ea:.2f} B={eb:.2f}  S=B-A={eb - ea:+.2f} '
          f'(per 30 scenes {30 * (eb - ea) / len(keys):+.2f})  one-sided p(B>A)={p:.4f}')
    changed = sum(1 for k in keys if A[k]['packed'] != B[k]['packed'])
    print(f'scenes with a different count: {changed}/{len(keys)}')

    def md(field):
        v = [B[k].get(field, 0.0) - A[k].get(field, 0.0) for k in keys
             if A[k].get(field) is not None and B[k].get(field) is not None]
        if len(v) < 2:
            return 'NA'
        return f'{st.mean(v):+.2f} (se {st.stdev(v) / math.sqrt(len(v)):.2f})'
    for f in ('packed', 'pct', 'fill', 'cog_score', 'placement_score', 'soft_item_score',
              'placement_contact', 'soft_contact', 'stability_score'):
        print(f'  d{f:16s} {md(f)}')

    seg = defaultdict(list)
    for k, dk in zip(keys, d):
        seg[_meta(A[k], scenes)].append(dk)
    print('segments (containers, shelf): S, one-sided p(worse)')
    for s, v in sorted(seg.items(), key=lambda x: str(x[0])):
        pw = perm_p([-x for x in v], args.perm)
        flag = '  <-- WORSE' if pw < 0.10 else ''
        print(f'  {str(s):14s} n={len(v):3d}  S={sum(v):+6.2f}  p(worse)={pw:.3f}{flag}')


if __name__ == '__main__':
    main()
