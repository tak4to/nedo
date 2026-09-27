"""W_SLIVER sweep, paired against the base re-run under the same load."""
import json, math, os, sys

def load(t):
    p = f'ab_{t}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def ms(d):
    n = len(d); m = sum(d) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

base_tag = sys.argv[1] if len(sys.argv) > 1 else 'SV_base'
arms = sys.argv[2:] or ['SV_400', 'SV_1500', 'SV_5000']
base = load(base_tag)
print(f'base {base_tag}: placed {sum(base[k]["packed"] for k in base)}  packed% {sum(base[k]["pct"] for k in base)/len(base):.2f}')
for tag in arms:
    r = load(tag)
    if r is None:
        print(f'{tag}: (missing)'); continue
    ks = sorted(set(r) & set(base))
    dp, dpse = ms([r[k]['pct'] - base[k]['pct'] for k in ks])
    df, dfse = ms([r[k]['fill'] - base[k]['fill'] for k in ks])
    dc, dcse = ms([comp(r[k]) - comp(base[k]) for k in ks])
    d = [r[k]['packed'] - base[k]['packed'] for k in ks]
    print(f'{tag:9s} placed {sum(r[k]["packed"] for k in ks):5d}  packed% {dp:+5.2f} ({dpse:4.2f})  '
          f'fill {df:+5.2f} ({dfse:4.2f})  comp {dc:+5.2f} ({dcse:4.2f})  win/loss {sum(x>0 for x in d)}/{sum(x<0 for x in d)}')
