"""P3 weight sweep, pooled over all 123 priority scenes (dev+holdout shown separately too)."""
import json, math, os

def load(t):
    p = f'ab_{t}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def ms(d):
    d = [x for x in d if x is not None]
    if len(d) < 2:
        return float('nan'), float('nan')
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))

base = load('PR_base')
DEVP = ('pool_', 'ktest_', 'dense_')
print(f"{'arm':9s} {'buried':>7s} {'depth':>6s} {'height':>7s} {'place':>6s} | "
      f"{'packed% vs base':>18s} {'fill vs base':>16s} | dev / holdout packed%")
for tag in ('PR_base', 'PR_h100', 'PR_h200', 'PR_h300', 'PR_high', 'PR_door', 'PR_both'):
    r = load(tag)
    if r is None:
        print(f'{tag:9s} (missing)'); continue
    ks = sorted(set(r) & set(base))
    bur = sum(r[k]['n_prio_buried'] for k in ks)
    dep, _ = ms([r[k].get('prio_depth') for k in ks])
    hei, _ = ms([r[k].get('prio_height') for k in ks])
    plc, _ = ms([r[k].get('placement_score') for k in ks])
    dp, dpse = ms([r[k]['pct'] - base[k]['pct'] for k in ks])
    df, dfse = ms([r[k]['fill'] - base[k]['fill'] for k in ks])
    dv, dvse = ms([r[k]['pct'] - base[k]['pct'] for k in ks if k.startswith(DEVP)])
    dh, dhse = ms([r[k]['pct'] - base[k]['pct'] for k in ks if not k.startswith(DEVP)])
    print(f'{tag:9s} {bur:7d} {dep:6.3f} {hei:7.3f} {plc:6.2f} | '
          f'{dp:+8.2f} ({dpse:4.2f})   {df:+7.2f} ({dfse:4.2f}) | {dv:+5.2f} / {dh:+5.2f}')
