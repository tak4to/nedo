"""Score ab_<tag>.json files with objective.py (local proxy, not LB)."""
import json, sys
import objective as o
for tag in sys.argv[1:]:
    rows = json.load(open(f'ab_{tag}.json'))
    print(f"{tag:10s} composite={sum(o.composite(r) for r in rows)/len(rows):6.2f} "
          f"smooth={o.smooth_objective(rows):6.2f} step(40/46/50)={o.objective(rows):6.2f} "
          f"packed={sum(r['pct'] for r in rows)/len(rows):5.1f}% below46={sum(1 for r in rows if r['pct']<46)} "
          f"minpct={min(r['pct'] for r in rows):4.1f} p10pct={sorted(r['pct'] for r in rows)[3]:4.1f}")
