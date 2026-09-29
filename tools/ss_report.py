"""OFFLINE_CONSTRUCT=1 (CON_*) vs agents/submit baselines, paired per scene."""
import json, math, os

def load(tag):
    p = f'ab_{tag}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

def ms(d):
    n = len(d); m = sum(d) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float('nan')), n

def report(title, pairs):
    if not pairs:
        print(f'## {title}: (missing)'); return
    print(f'## {title}  n={len(pairs)}  (diff = SS - base)')
    for lab, f in (('placed', lambda r: r['packed']), ('packed %', lambda r: r['pct']),
                   ('fill', lambda r: r['fill']), ('composite', comp)):
        d = [f(a) - f(b) for a, b in pairs]
        m, se, n = ms(d)
        print(f'  {lab:10s} CON={sum(f(a) for a,_ in pairs)/n:7.2f} base={sum(f(b) for _,b in pairs)/n:7.2f} '
              f'diff={m:+6.2f} (se {se:4.2f})  win/loss={sum(x>0 for x in d)}/{sum(x<0 for x in d)}')

con = load('SS_suite'); our = load('OUR_suite'); pub = load('PUB_suite')
if con:
    opt = [k for k in con if json.load(open('scenes_pubsuite.json'))[k]['agent']['optimize']]
    report('public suite, optimize scenes (vs OUR)', [(con[k], our[k]) for k in opt])
    report('public suite, all 26 (vs OUR)', [(con[k], our[k]) for k in con])
    print(f"  placed totals (26): CON={sum(r['packed'] for r in con.values())} "
          f"OUR={sum(r['packed'] for r in our.values())} PUB={sum(r['packed'] for r in pub.values())}")
c44 = load('SS_opt44')
if c44:
    base = {}
    for s in ['pool', 'test', 'ktest', 'shelftest']:
        for k, r in load(f'OUR_{s}').items():
            base[f'{s}_{k}'] = r
    report('catalog optimize scenes, own look_ahead (vs OUR_<set>)', [(c44[k], base[k]) for k in c44])
ck1 = load('SS_basek1'); bk1 = load('UB_basek1')
if ck1:
    report('catalog optimize scenes as official A, k=1 (vs UB_basek1)', [(ck1[k], bk1[k]) for k in ck1])
