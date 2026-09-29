"""How many soft / priority bags each tag loads, optimize vs online scenes.

hotfix6 stopped leaving priority bags behind and the leaderboard moved
+3.10 (2026-09-28) -- the first move in eight submissions -- so the official
placement score counts a bag left behind as a failure. Soft bags are scored
the same way (README:383) and optimize()'s sort keys all put them last.
This reports, per tag and per scene kind (optimize = opt_s > 0):

  soft loaded %   (soft_all_column * n_soft_total / 100 + n_soft_bad) / n_soft_total
  other loaded %  non-soft bags loaded / non-soft bags
  soft_all        mean soft_all_column (loaded and not buried, over all soft)
  prio_all        mean prio_all_column
  packed          bags loaded / bags

    python soft_audit.py TAG [TAG ...]      # reads ab_<TAG>_<pool>.json
    python soft_audit.py --files a.json b.json   # arbitrary result files
"""
import json
import sys

POOLS = ('pool', 'test', 'ktest', 'shelftest')


def rows_of(tag):
    out = []
    for p in POOLS:
        try:
            out += json.load(open(f'ab_{tag}_{p}.json'))
        except FileNotFoundError:
            pass
    return out


def summarize(label, rows):
    rows = [r for r in rows if 'n_soft_total' in r]
    for kind in ('optimize', 'online'):
        R = [r for r in rows if (r.get('opt_s', 0) > 0) == (kind == 'optimize')]
        if not R:
            continue
        n_soft = sum(r['n_soft_total'] for r in R)
        soft_loaded = sum(r['soft_all_column'] * r['n_soft_total'] / 100.0 + r.get('n_soft_bad', 0)
                          for r in R)
        n_other = sum(r['total'] for r in R) - n_soft
        other_loaded = sum(r['packed'] for r in R) - soft_loaded
        n = len(R)
        print(f'{label:10s} {kind:8s} n={n:3d}  soft {n_soft:5d} bags, loaded '
              f'{100 * soft_loaded / max(n_soft, 1):5.1f}%  | other loaded '
              f'{100 * other_loaded / max(n_other, 1):5.1f}%  | soft_all '
              f'{sum(r["soft_all_column"] for r in R) / n:5.1f}  prio_all '
              f'{sum(r["prio_all_column"] for r in R) / n:5.1f}  packed '
              f'{sum(r["packed"] for r in R) / max(sum(r["total"] for r in R), 1):.3f}  fill '
              f'{sum(r["fill"] for r in R) / n:5.2f}')


PAIR_FIELDS = ('soft_all_column', 'soft_all_contact', 'prio_all_column', 'packed', 'pct',
               'fill', 'fill_loose', 'cog_score', 'stability_score', 'placement_score',
               'soft_item_score')


def pair(base, treats):
    """Scene-paired deltas (treat - base) on ab_<TAG>.json files, overall and
    per scene group (the part of the name before ':'), plus the number of
    scenes that ended below 50% of their bags (the collapse count)."""
    import math
    import statistics as st
    from xkpi import perm_p
    A = {r['scene']: r for r in json.load(open(f'ab_{base}.json'))}
    for t in treats:
        B = {r['scene']: r for r in json.load(open(f'ab_{t}.json'))}
        ks = sorted(set(A) & set(B))
        print(f'== {base} -> {t}  (n={len(ks)})')
        for f in PAIR_FIELDS:
            v = [B[k][f] - A[k][f] for k in ks if f in A[k] and f in B[k]]
            if len(v) > 1:
                print(f'   d{f:17s} {st.mean(v):+7.2f} (se {st.stdev(v) / math.sqrt(len(v)):5.2f})'
                      f'  p(>0)={perm_p(v, 20000):.3f}')
        print(f'   below 50% of bags: {sum(A[k]["pct"] < 50 for k in ks)} -> '
              f'{sum(B[k]["pct"] < 50 for k in ks)}')
        groups = sorted({k.split(':')[0] for k in ks})
        for g in groups:
            gk = [k for k in ks if k.split(':')[0] == g]
            d = lambda f: st.mean(B[k][f] - A[k][f] for k in gk)  # noqa: E731
            print(f'   [{g:9s} n={len(gk):2d}] soft_all {d("soft_all_column"):+6.1f}  prio_all '
                  f'{d("prio_all_column"):+6.1f}  packed {d("packed"):+5.2f}  fill {d("fill"):+5.2f}'
                  f'  cog {d("cog_score"):+5.2f}  stab {d("stability_score"):+5.2f}')


def main():
    args = sys.argv[1:]
    if args and args[0] == '--pair':
        pair(args[1], args[2:])
        return
    if args and args[0] == '--files':
        for f in args[1:]:
            summarize(f.replace('ab_', '').replace('.json', ''), json.load(open(f)))
        return
    for tag in args:
        summarize(tag, rows_of(tag))


if __name__ == '__main__':
    main()
