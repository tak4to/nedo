"""P3 report: priority baggage retrievability vs packing cost.
prio_depth 0 = at the door, 1 = back wall. prio_height 1 = at the top of the stack."""
import json, math, os

def load(t):
    p = f'ab_{t}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def ms(d):
    d = [x for x in d if x is not None]
    if len(d) < 2:
        return (float('nan'), float('nan'), len(d))
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)), len(d)

DEV = ('pool_', 'ktest_', 'dense_')
base = load('PR_base')
for grp, sel in (('dev (pool/ktest/dense)', lambda k: k.startswith(DEV)),
                 ('holdout (test/shelftest/densetest)', lambda k: not k.startswith(DEV))):
    ks = [k for k in base if sel(k)]
    print(f'== {grp}  n={len(ks)}')
    for tag in ('PR_base', 'PR_door', 'PR_high', 'PR_both'):
        r = load(tag)
        if r is None:
            print(f'  {tag}: (missing)'); continue
        ks2 = [k for k in ks if k in r]
        f = lambda key: [r[k].get(key) for k in ks2]
        depth, _, _ = ms(f('prio_depth')); height, _, _ = ms(f('prio_height'))
        bur = sum(r[k]['n_prio_buried'] for k in ks2)
        plc = sum(r[k].get('placement_score', 0) for k in ks2) / len(ks2)
        dp, dpse, _ = ms([r[k]['pct'] - base[k]['pct'] for k in ks2])
        df, dfse, _ = ms([r[k]['fill'] - base[k]['fill'] for k in ks2])
        print(f'  {tag:8s} prio_depth={depth:.3f} prio_height={height:.3f} buried={bur:3d} '
              f'placement={plc:5.2f} | vs base packed% {dp:+5.2f} ({dpse:.2f}) fill {df:+5.2f} ({dfse:.2f})')
