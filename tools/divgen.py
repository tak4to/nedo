"""Scenes with graded item diversity (plan v3, option 1: 2026-09-26).

S1 (small-first seed orders in optimize) was gated at "distinct shapes /
items >= 0.5": the catalogue scenes sit at 0.06-0.17 and the public
suite / scenes_div44 at 1.00, and nothing in between was ever measured, so
whether S1 helps or hurts at, say, 0.3 was simply unknown. Here each scene
draws its items from a palette of M random types (dims as scenes_div44:
length 0.40-0.80, width 0.30-0.60, height 0.18-0.45), so diversity ~ M/n.

    python divgen.py [--seed 0] [--per 12] [--out scenes_divmix.json]
"""
import argparse
import json
import random

from scenegen import REAL_GEOMS, _item, make_scene

LEVELS = (7, 12, 18, 27, 45)    # palette sizes; n = 45 items -> 0.16 .. 1.0


def build(seed=0, per=12, n_items=45, k=1, offline=True):
    rng = random.Random(seed)
    out = {}
    for m in LEVELS:
        for i in range(per):
            shelf = i % 2 == 1
            cfg = make_scene(seed=rng.randrange(1 << 30), n_containers=1, n_items=n_items,
                             shelf=shelf, look_ahead=k, offline=offline,
                             geom=REAL_GEOMS[1 if shelf else 0])
            palette = []
            for _ in range(m):
                l, w, h = rng.uniform(0.40, 0.80), rng.uniform(0.30, 0.60), rng.uniform(0.18, 0.45)
                vol = l * w * h
                palette.append(dict(length=round(l, 3), width=round(w, 3), height=round(h, 3),
                                    mass=round(max(2.0, 155.0 * vol * rng.uniform(0.7, 1.3)), 1),
                                    is_soft=rng.random() < 0.3))
            # every type at least once when m <= n, the rest uniform
            picks = list(range(m)) if m <= n_items else []
            picks += [rng.randrange(m) for _ in range(n_items - len(picks))]
            rng.shuffle(picks)
            items = [_item(j, palette[t], rng.random() < 0.1) for j, t in enumerate(picks)]
            cfg['item_stream']['item_list'] = items
            cfg['item_stream']['visible_pool'] = items[:cfg['item_stream']['look_ahead']]
            div = len({tuple(sorted((it['length'], it['width'], it['height']))) for it in items}) / len(items)
            out[f'M{m:02d}_{i:02d}_{"S" if shelf else "N"}'] = cfg
            cfg['_diversity'] = round(div, 3)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per', type=int, default=12)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--online', action='store_true')
    ap.add_argument('--out', default='scenes_divmix.json')
    a = ap.parse_args()
    sc = build(a.seed, a.per, k=a.k, offline=not a.online)
    json.dump(sc, open(a.out, 'w'))
    from collections import defaultdict
    d = defaultdict(list)
    for k, v in sc.items():
        d[k[:3]].append(v['_diversity'])
    print('wrote', len(sc), 'scenes;', {k: round(sum(v) / len(v), 3) for k, v in sorted(d.items())})
