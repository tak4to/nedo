"""H-slow report: budgets scaled x0.3 / x0.1 (slower-machine emulation) vs x1.0,
paired over all 128 scenes (4 sets). Also SL10 vs the E1 baselines as a sanity
check that x1.0 reproduces the shipped agent."""
import json, math, os, collections

SETS = ['pool', 'ktest', 'shelftest', 'test']
OLD = {'pool': 'G_base', 'ktest': 'B_ktest', 'shelftest': 'B_shelftest', 'test': 'B_test'}

def load(tag):
    p = f'ab_{tag}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

def cause(st):
    return 'stuck' if st.get('is_valid') is False else ('topple' if st.get('is_placed_safe') is False else 'done')

def ms(d):
    n = len(d); m = sum(d) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float('nan')), n

def pair(a_tag, b_tag_of):
    rows = []
    for s in SETS:
        a, b = load(f'{a_tag}_{s}'), load(b_tag_of(s))
        if a is None or b is None:
            continue
        rows += [(a[k], b[k]) for k in sorted(set(a) & set(b))]
    return rows

for arm in ('SL03', 'SL01'):
    rows = pair(arm, lambda s: f'SL10_{s}')
    if not rows:
        continue
    print(f'### {arm} vs SL10 (n={len(rows)})')
    for lab, f in (('packed %', lambda r: r['pct']), ('fill', lambda r: r['fill']), ('composite', comp)):
        m, se, n = ms([f(a) - f(b) for a, b in rows])
        print(f'  {lab:10s} {arm}={sum(f(a) for a, _ in rows)/n:6.2f} SL10={sum(f(b) for _, b in rows)/n:6.2f} diff={m:+6.2f} (se {se:4.2f})')
    print('  end cause', arm, collections.Counter(cause(a['status']) for a, _ in rows),
          'SL10', collections.Counter(cause(b['status']) for _, b in rows))
    by = collections.defaultdict(list)
    for a, b in rows:
        by[a['total'] > 60].append(a['pct'] - b['pct'])
    for big, d in sorted(by.items()):
        print(f"  {'2 containers-ish (>60 items)' if big else '<=60 items':28s} n={len(d):3d} d_packed={sum(d)/len(d):+6.2f}")

rows = pair('SL10', lambda s: OLD[s])
if rows:
    m, se, n = ms([a['pct'] - b['pct'] for a, b in rows])
    print(f'### sanity SL10 vs E1 baselines: packed diff={m:+.2f} (se {se:.2f}) n={n}')
