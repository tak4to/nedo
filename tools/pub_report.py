"""Head-to-head report: public solver (PUB_*) vs agents/submit (OUR_*), paired
per scene over the 4 sets, broken down by scene type. Paired differences are
PUB - OUR (positive = the public solver is better)."""
import json, math, os, collections, sys

SETS = ['pool', 'ktest', 'shelftest', 'test']
A, B = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('PUB', 'OUR')

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

rows = []
for s in SETS:
    a, b = load(f'{A}_{s}'), load(f'{B}_{s}')
    if a is None or b is None:
        continue
    cfgs = json.load(open(f'scenes_{s}.json'))
    for k in sorted(set(a) & set(b)):
        c = cfgs[k]; cl = c['containers']['container_list']
        attr = dict(
            task=('A(opt)' if c['agent']['optimize'] else ('C(k=1)' if c['item_stream']['look_ahead'] == 1 else 'B(pool)')),
            conts=f"{len(cl)}cont",
            shelf='shelf' if any(x.get('require_shelf') for x in cl) else 'noshelf',
            prepacked='prepacked' if any(x.get('packed_items') for x in cl) else 'empty',
            prio='prio-cont' if any(x.get('is_prioritized') for x in cl) else 'no-prio-cont')
        rows.append((s, k, a[k], b[k], attr))

print(f'### {A} vs {B}  n={len(rows)} scenes  (diff = {A} - {B})')
for lab, f in (('packed %', lambda r: r['pct']), ('fill', lambda r: r['fill']),
               ('cog', lambda r: r.get('cog_score', 0)), ('stability', lambda r: r.get('stability_score', 0)),
               ('placement', lambda r: r.get('placement_score', 0)), ('soft', lambda r: r.get('soft_item_score', 0)),
               ('composite', comp)):
    d = [f(a) - f(b) for _, _, a, b, _ in rows]
    m, se, n = ms(d)
    print(f'  {lab:10s} {A}={sum(f(a) for _,_,a,_,_ in rows)/n:6.2f} {B}={sum(f(b) for _,_,_,b,_ in rows)/n:6.2f} '
          f'diff={m:+6.2f} (se {se:4.2f})  win/loss={sum(x>0 for x in d)}/{sum(x<0 for x in d)}')
print('  end cause', A, dict(collections.Counter(cause(a['status']) for _,_,a,_,_ in rows)),
      B, dict(collections.Counter(cause(b['status']) for _,_,_,b,_ in rows)))
qs = lambda v: [sorted(v)[int(q * (len(v) - 1))] for q in (0.1, 0.25, 0.5, 0.75)]
print(f"  packed% quantiles p10/p25/p50/p75: {A}={qs([a['pct'] for _,_,a,_,_ in rows])} "
      f"{B}={qs([b['pct'] for _,_,_,b,_ in rows])}")
print(f"  max policy s: {A}={max(a['max_policy_s'] for _,_,a,_,_ in rows):.2f} {B}={max(b['max_policy_s'] for _,_,_,b,_ in rows):.2f}")

for dim in ('task', 'conts', 'shelf', 'prepacked', 'prio'):
    g = collections.defaultdict(list)
    for _, _, a, b, at in rows:
        g[at[dim]].append((a, b))
    print(f'  -- by {dim}')
    for key, v in sorted(g.items()):
        dp = [a['pct'] - b['pct'] for a, b in v]; dc = [comp(a) - comp(b) for a, b in v]
        m, se, n = ms(dp) if len(dp) > 1 else (dp[0], float('nan'), 1)
        mc = sum(dc) / len(dc)
        print(f'     {key:12s} n={n:3d}  packed {A}={sum(a["pct"] for a,_ in v)/n:5.1f} {B}={sum(b["pct"] for _,b in v)/n:5.1f} '
              f'd={m:+6.2f} (se {se:4.2f})  composite d={mc:+5.2f}')
