"""Paired comparison on the COMPOSITE (5-metric mean), which is what the
leaderboard actually scores -- LB 53.4 against a local composite of 57.3,
versus a fill of 23. Also prints packed%, because the composite can be
gamed by placing almost nothing (SUPPORT_MIN_COVER=1.0 scores 60.7 at
31.7% placed) and the README's "minimum items" threshold sits somewhere
between 42% and 46% placed -- below it, everything but fill scores zero."""
import json, sys, math
K = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']
def load(t):
    return {r['scene']: r for r in json.load(open(f'ab_{t}.json'))}
def comp(r):
    return sum(r.get(k, 0) for k in K) / 5.0
bases = [load(t) for t in sys.argv[1].split(',')]
scenes = sorted(set.intersection(*[set(b) for b in bases]))
def bmean(s, f):
    return sum(f(b[s]) for b in bases) / len(bases)
for tag in sys.argv[2].split(','):
    a = load(tag)
    ss = [s for s in scenes if s in a]
    d = [(comp(a[s]) - bmean(s, comp), s) for s in ss]
    n = len(d); m = sum(x for x, _ in d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x, _ in d) / (n - 1)) if n > 1 else 0.0
    w = sum(1 for x, _ in d if x > 1e-9); l = sum(1 for x, _ in d if x < -1e-9)
    dp = sum(a[s]['pct'] - bmean(s, lambda r: r['pct']) for s in ss) / n
    parts = ' '.join(f'{k.split("_")[0]}{sum(a[s].get(k,0) for s in ss)/n - sum(bmean(s, lambda r,k=k: r.get(k,0)) for s in ss)/n:+.2f}' for k in K)
    print(f'{tag:14s} n={n} dCOMPOSITE={m:+6.2f} (se {sd/math.sqrt(n):.2f}) W/L={w}/{l}  dpacked={dp:+5.2f}')
    # Threshold model: below the README's "minimum items" cliff only fill
    # counts, so the score is a step function and packed% matters far more
    # than the composite suggests. The cliff sits between 42% and 46%
    # placed (LB 27.27 at 42.3%, 53.4 at 46.5%); sweep it to check the
    # conclusion does not depend on where exactly it is.
    for TH in (40, 46, 50):
        f = lambda r: r['fill'] if r['pct'] < TH else comp(r)
        base = sum(bmean(s, f) for s in ss) / n
        new = sum(f(a[s]) for s in ss) / n
        cross = sum(1 for s in ss if bmean(s, lambda r: r['pct']) < TH <= a[s]['pct'])
        fell = sum(1 for s in ss if a[s]['pct'] < TH <= bmean(s, lambda r: r['pct']))
        print(f'                thresh {TH}%: estLB {base:5.2f} -> {new:5.2f} ({new-base:+5.2f})  crossed+{cross}/-{fell}')
    print(f'                {parts}')
    d.sort()
    print('   worst:', ' '.join(f'{s}{x:+.1f}' for x, s in d[:4]))
    print('   best :', ' '.join(f'{s}{x:+.1f}' for x, s in d[-4:]))
