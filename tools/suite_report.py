"""Per-scene table for the public 26-scene suite: PUB_suite vs OUR_suite."""
import json, math

def load(tag):
    return {r['scene']: r for r in json.load(open(f'ab_{tag}.json'))}

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

def cause(st):
    return 'stuck' if st.get('is_valid') is False else ('topple' if st.get('is_placed_safe') is False else 'done')

a, b = load('PUB_suite'), load('OUR_suite')
print(f"{'scene':30s} {'PUB placed':>10s} {'OUR placed':>10s}   {'PUB fill':>8s} {'OUR fill':>8s}  PUB end / OUR end")
groups = {}
for k in sorted(a):
    ra, rb = a[k], b[k]
    print(f"{k:30s} {ra['packed']:4d}/{ra['total']:<4d}  {rb['packed']:4d}/{rb['total']:<4d}   {ra['fill']:8.2f} {rb['fill']:8.2f}  {cause(ra['status']):6s} / {cause(rb['status'])}")
    groups.setdefault(k[0], []).append((ra, rb))
print()
for g, v in sorted(groups.items()):
    n = len(v)
    print(f"group {g}: n={n}  placed PUB={sum(x['packed'] for x,_ in v)} OUR={sum(y['packed'] for _,y in v)}  "
          f"pct PUB={sum(x['pct'] for x,_ in v)/n:5.1f} OUR={sum(y['pct'] for _,y in v)/n:5.1f}  "
          f"comp PUB={sum(comp(x) for x,_ in v)/n:5.2f} OUR={sum(comp(y) for _,y in v)/n:5.2f}")
n = len(a)
print(f"ALL: placed PUB={sum(r['packed'] for r in a.values())} OUR={sum(r['packed'] for r in b.values())}  "
      f"comp PUB={sum(comp(r) for r in a.values())/n:5.2f} OUR={sum(comp(r) for r in b.values())/n:5.2f}")
