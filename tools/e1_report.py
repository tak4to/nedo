"""E1 report: each condition vs the shipped agent, on 4 scene sets.

Baselines (agents/submit): G_base (pool), B_ktest, B_shelftest, B_test.
Criterion: mean items-placed % over ALL scenes of the 3 UNSEEN sets (paired,
se) and composite. Never condition on the baseline's outcome: scenes the
baseline ended 'topple' regress up under any change, and scenes it ended
'stuck' regress down (docs/2026-09-22-構造的な穴の探索.md §5) -- the
'stuck' column is shown for reference only and is biased negative.
"""
import json, math, os, sys
import objective as o

SETS = [('pool', 'G_base'), ('ktest', 'B_ktest'), ('shelftest', 'B_shelftest'), ('test', 'B_test')]
UNSEEN = ('ktest', 'shelftest', 'test')


def load(tag):
    p = f'ab_{tag}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None


def cause(st):
    return 'stuck' if st['is_valid'] is False else ('topple' if st['is_placed_safe'] is False else 'done')


def stat(d):
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n) if n > 1 else float('nan')
    return m, se


def report(cond):
    rows = []
    unseen_dp, unseen_dc, stuck_dp = [], [], []
    for s, base in SETS:
        b = load(base); c = load(f'{cond}_{s}')
        if b is None or c is None:
            continue
        sc = sorted(set(b) & set(c))
        dp = [c[k]['pct'] - b[k]['pct'] for k in sc]
        dc = [o.composite(c[k]) - o.composite(b[k]) for k in sc]
        st = [c[k]['pct'] - b[k]['pct'] for k in sc if cause(b[k]['status']) == 'stuck']
        tb = sum(1 for k in sc if cause(b[k]['status']) == 'topple')
        tc = sum(1 for k in sc if cause(c[k]['status']) == 'topple')
        m, se = stat(dp)
        rows.append(f"   {s:10s} dpct={m:+5.2f} (se {se:4.2f})  dcomp={sum(dc)/len(dc):+5.2f}  "
                     f"stuck(biased-) dpct={stat(st)[0]:+5.2f}  topple {tb}->{tc}  "
                     f"step {o.objective(list(b[k] for k in sc)):5.2f}->{o.objective(list(c[k] for k in sc)):5.2f}")
        if s in UNSEEN:
            unseen_dp += dp; unseen_dc += dc; stuck_dp += st
    if not rows:
        return
    print(f"== {cond}")
    print("\n".join(rows))
    if unseen_dp:
        m, se = stat(unseen_dp)
        verdict = 'ADOPT?' if (m >= 2.0 and se < 1.0 and sum(unseen_dc) / len(unseen_dc) >= 0) else '-'
        print(f"   UNSEEN({len(unseen_dp)}) dpct={m:+5.2f} (se {se:4.2f})  dcomp={sum(unseen_dc)/len(unseen_dc):+5.2f}  "
              f"stuck(biased-) dpct={stat(stuck_dp)[0]:+5.2f} (n={len(stuck_dp)})   {verdict}")


if __name__ == '__main__':
    for c in sys.argv[1:]:
        report(c)
