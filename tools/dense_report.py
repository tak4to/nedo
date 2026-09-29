"""How the shipped agent behaves at the real density (50-60 bags/container)
vs the 40-50 pools it was developed on."""
import collections, json, math, os

def load(t):
    p = f'ab_{t}.json'
    return {r['scene']: r for r in json.load(open(p))} if os.path.exists(p) else None

def cause(st):
    return 'stuck' if st.get('is_valid') is False else ('topple' if st.get('is_placed_safe') is False else 'done')

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

def show(tag, scenes_file):
    r = load(tag)
    if r is None:
        print(f'{tag}: (missing)'); return
    sc = json.load(open(scenes_file))
    n = len(r)
    per_c = lambda k: r[k]['packed'] / len(sc[k]['containers']['container_list'])
    print(f"{tag:9s} n={n}  placed/scene={sum(r[k]['packed'] for k in r)/n:5.1f}  "
          f"placed/container={sum(per_c(k) for k in r)/n:5.1f}  packed%={sum(r[k]['pct'] for k in r)/n:5.1f}  "
          f"fill={sum(r[k]['fill'] for k in r)/n:5.2f}  comp={sum(comp(r[k]) for k in r)/n:5.2f}  "
          f"end={dict(collections.Counter(cause(r[k]['status']) for k in r))}")
    for lab, sel in (('1 container', lambda k: len(sc[k]['containers']['container_list']) == 1),
                     ('2 containers', lambda k: len(sc[k]['containers']['container_list']) == 2),
                     ('shelf', lambda k: any(c['require_shelf'] for c in sc[k]['containers']['container_list'])),
                     ('no shelf', lambda k: not any(c['require_shelf'] for c in sc[k]['containers']['container_list'])),
                     ('offline (A)', lambda k: sc[k]['agent']['optimize']),
                     ('online (B/C)', lambda k: not sc[k]['agent']['optimize'])):
        ks = [k for k in r if sel(k)]
        if ks:
            print(f"    {lab:13s} n={len(ks):2d}  placed/container={sum(per_c(k) for k in ks)/len(ks):5.1f}  "
                  f"fill={sum(r[k]['fill'] for k in ks)/len(ks):5.2f}  "
                  f"end={dict(collections.Counter(cause(r[k]['status']) for k in ks))}")

show('DN_dev', 'scenes_dense.json')
show('DN_test', 'scenes_densetest.json')
print('--- reference: the 40-50 pools (same agent, same budgets)')
for tag, f in (('OUR_pool', 'scenes_pool.json'), ('OUR_test', 'scenes_test.json'),
               ('OUR_ktest', 'scenes_ktest.json'), ('OUR_shelftest', 'scenes_shelftest.json')):
    r = load(tag)
    if r is None:
        continue
    sc = json.load(open(f)); n = len(r)
    per_c = sum(r[k]['packed'] / len(sc[k]['containers']['container_list']) for k in r) / n
    print(f"{tag:9s} n={n}  placed/container={per_c:5.1f}  packed%={sum(r[k]['pct'] for k in r)/n:5.1f}  "
          f"fill={sum(r[k]['fill'] for k in r)/n:5.2f}  comp={sum(comp(r[k]) for k in r)/n:5.2f}  "
          f"end={dict(collections.Counter(cause(r[k]['status']) for k in r))}")
