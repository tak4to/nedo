"""Per-diversity-level paired report for scenes_divmix-style A/Bs.

    python div_report.py ab_BASE.json ab_TREAT.json [--scenes scenes_divmix.json]
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict

from xkpi import F, perm_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--scenes', default='scenes_divmix.json')
    args = ap.parse_args()
    A = {r['scene']: r for r in json.load(open(args.a))}
    B = {r['scene']: r for r in json.load(open(args.b))}
    sc = json.load(open(args.scenes))
    groups = defaultdict(list)
    for k in sorted(set(A) & set(B)):
        lvl = f"{k[:3]} div={sc[k].get('_diversity', 0):.2f}"
        groups[lvl].append(k)
        groups[f"{k[:3]} shelf={k.endswith('S')}"].append(k)

    def row(name, ks):
        dp = [B[k]['packed'] - A[k]['packed'] for k in ks]
        df = [B[k]['fill'] - A[k]['fill'] for k in ks]
        dx = [F(B[k]['pct'], B[k]['packed']) - F(A[k]['pct'], A[k]['packed']) for k in ks]
        se = st.stdev(dp) / math.sqrt(len(dp)) if len(dp) > 1 else float('nan')
        w = sum(1 for d in dp if d > 0)
        l = sum(1 for d in dp if d < 0)
        print(f'  {name:22s} n={len(ks):2d}  placed {st.mean([A[k]["packed"] for k in ks]):5.1f} -> '
              f'{st.mean([B[k]["packed"] for k in ks]):5.1f}  d={st.mean(dp):+5.2f} (se {se:4.2f}) '
              f'{w}W/{l}L  dfill={st.mean(df):+5.2f}  dcross={sum(dx):+5.2f}  p={perm_p(dx, 5000):.3f}')
    for g in sorted(groups):
        if 'div=' in g:
            row(g, groups[g])
    print('  ---')
    row('ALL', sorted(set(A) & set(B)))


if __name__ == '__main__':
    main()
