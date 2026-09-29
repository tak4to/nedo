"""Ablation report on scenes_div44: each arm vs DIV_our (and vs DIV_pub for PUB ablations)."""
import json, math, os
def load(t):
    p = f'ab_{t}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None
def ms(d):
    n = len(d); m = sum(d) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
sc = json.load(open('scenes_div44.json'))
one = [k for k in sc if len(sc[k]['containers']['container_list']) == 1]
our, pub = load('DIV_our'), load('DIV_pub')
for tag in ('DIV_our', 'DIV_pub', 'DIV_ours', 'DIV_pub_oc0', 'DIV_pub_pe0', 'DIV_pub_lr0'):
    r = load(tag)
    if r is None:
        print(f'{tag:12s} (missing)'); continue
    ks = sorted(r)
    m, se = ms([r[k]['pct'] - our[k]['pct'] for k in ks]); mf, sf = ms([r[k]['fill'] - our[k]['fill'] for k in ks])
    line = (f"{tag:12s} placed={sum(r[k]['packed'] for k in ks):5d} (1cont {sum(r[k]['packed'] for k in one):4d})  "
            f"vs OUR packed% {m:+6.2f} ({se:.2f}) fill {mf:+6.2f} ({sf:.2f})")
    if tag.startswith('DIV_pub_'):
        m2, se2 = ms([r[k]['pct'] - pub[k]['pct'] for k in ks])
        line += f"   vs PUB packed% {m2:+6.2f} ({se2:.2f})"
    print(line)
