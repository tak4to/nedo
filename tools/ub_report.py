"""Upper-bound report: official-A (LNS order, look_ahead=1) vs free choice among 40.
Paired per scene over all 44 optimize scenes; never condition on either arm's outcome."""
import json, math, collections

def load(tag):
    return {r['scene']: r for r in json.load(open(f'ab_{tag}.json'))}

def comp(r):
    return (r['fill'] + r.get('cog_score', 0) + r.get('stability_score', 0)
            + r.get('placement_score', 0) + r.get('soft_item_score', 0)) / 5.0

def cause(st):
    if not isinstance(st, dict):
        return str(st)
    return 'stuck' if st.get('is_valid') is False else ('topple' if st.get('is_placed_safe') is False else 'done')

def ms(d):
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float('nan')
    return m, se

b, c = load('UB_basek1'), load('UB_choice')
sc = sorted(set(b) & set(c))
for key, f in (('packed %', lambda r: r['pct']), ('fill', lambda r: r['fill']), ('composite', comp)):
    d = [f(c[k]) - f(b[k]) for k in sc]
    m, se = ms(d)
    wins = sum(x > 0 for x in d); losses = sum(x < 0 for x in d)
    print(f'{key:10s} base={sum(f(b[k]) for k in sc)/len(sc):6.2f} choice={sum(f(c[k]) for k in sc)/len(sc):6.2f} '
          f'diff={m:+6.2f} (se {se:4.2f})  win/loss={wins}/{losses}  n={len(sc)}')
print('end cause base  ', collections.Counter(cause(b[k]['status']) for k in sc))
print('end cause choice', collections.Counter(cause(c[k]['status']) for k in sc))
by = collections.defaultdict(list)
for k in sc:
    by[(b[k]['total'], len(json.load(open('scenes_UB_base.json'))[k]['containers']['container_list']))].append(c[k]['pct'] - b[k]['pct'])
for g, d in sorted(by.items()):
    print(f'  items={g[0]:3d} conts={g[1]}  n={len(d):2d}  d_packed={sum(d)/len(d):+6.2f}')
print('max policy s (choice):', max(c[k]['max_policy_s'] for k in sc))
