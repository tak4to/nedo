"""Paired comparison of two tags over the four catalogue pools.

Reports the mean difference (B - A) of every score, and the equal-weight
mean of the five scores (fill, cog, stability, placement, soft; column rule
as in scores.placement_scores) with a one-sided sign-flip permutation p.
Past the item threshold -- where 2026-09-27's analysis puts us -- the
continuous scores are what the leaderboard moves with, not crossings.

    python cmp5.py BASE_TAG TREAT_TAG [TREAT_TAG ...]    # reads ab_<TAG>_<pool>.json
"""
import json
import math
import statistics as st
import sys

from xkpi import perm_p

POOLS = ('pool', 'test', 'ktest', 'shelftest')
FIELDS = ('packed', 'fill', 'cog_score', 'stability_score', 'placement_score',
          'soft_item_score', 'placement_contact', 'soft_contact')


def load(tag):
    out = {}
    for p in POOLS:
        for r in json.load(open(f'ab_{tag}_{p}.json')):
            out[f'{p}:{r["scene"]}'] = r
    return out


def comp(r):
    return (r['fill'] + r['cog_score'] + r['stability_score'] + r['placement_score']
            + r['soft_item_score']) / 5.0


def main():
    base, treats = sys.argv[1], sys.argv[2:]
    A = load(base)
    for t in treats:
        B = load(t)
        ks = sorted(set(A) & set(B))
        print(f'== {base} -> {t}  (n={len(ks)})')
        for f in FIELDS:
            v = [B[k][f] - A[k][f] for k in ks if f in A[k] and f in B[k]]
            if len(v) > 1:
                print(f'   d{f:18s} {st.mean(v):+6.2f} (se {st.stdev(v) / math.sqrt(len(v)):4.2f})')
        dc = [comp(B[k]) - comp(A[k]) for k in ks]
        print(f'   equal-weight 5-score mean {st.mean(dc):+.2f} (se {st.stdev(dc) / math.sqrt(len(dc)):.2f}) '
              f'p(B>A)={perm_p(dc, 20000):.3f}')


if __name__ == '__main__':
    main()
