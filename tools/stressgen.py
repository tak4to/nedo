"""Stress scenes: configurations the local pools never exercise.

Every local pool used buffer=0.0, max_space=1, at most two containers of the
two shipped geometries and the seven catalogue items. README:460 says the
scored scenes differ in container count/size and in item types, counts and
order, and README:163 documents `buffer` (the simulator default is 0.01).
A scene outside the pools can end the episode far below the item threshold,
which no amount of average-case tuning shows. These families look for that:

  buf    buffer 0.005-0.04 on both shipped geometries
  geo    other ULD-like sizes (L 1.56-2.44, W 1.19-1.53, H 1.14-1.63),
         thickness 0.02-0.05, chamfer cut_x 0.30-0.88 / cut_y 0.30-0.65
  multi  3-4 containers, mixed geometries, a priority container
  pre    pre-packed items (0-40% of the volume), back floor or door side
  item   catalogue +-20% with many small items, long items (golf bag,
         stroller, skis), mass 2-32 kg, soft 10-80%, priority 0-30%
  flow   look_ahead x max_space {1, k/2, k}, big items late, priority
         clustered, soft first

Ranges follow the airport-side review in the 2026-09-26 plan. Usage:
    python stressgen.py [--seed 0] [--out scenes_stress.json]
"""
import argparse
import copy
import json
import math
import random

from scenegen import CATALOG, CATALOG_WEIGHTS, REAL_GEOMS, _item, make_scene


def _geom(rng, shelf):
    """A plausible ULD-like container that still leaves the notch strip and
    the shelf band physically meaningful."""
    H = rng.uniform(1.14, 1.63)
    L = rng.uniform(1.56, 2.44)
    W = rng.uniform(1.19, 1.53)
    t = rng.uniform(0.02, 0.05)
    cut_x = rng.uniform(0.30, min(0.88, L / 2.0 - 0.2))
    cut_y = rng.uniform(0.30, min(0.65, H / 2.0 - 0.1))
    return dict(L=round(L, 3), W=round(W, 3), H=round(H, 3), thickness=round(t, 3),
                cut_x=round(cut_x, 3), cut_y=round(cut_y, 3))


def _set_container(c, g, buffer, shelf, prio):
    c.update(length=g['L'], width=g['W'], height=g['H'], thickness=g['thickness'],
             cut_x=g['cut_x'], cut_y=g['cut_y'], buffer=buffer,
             require_shelf=shelf, is_prioritized=prio)


def _inner_volume(c):
    return ((c['length'] - 2 * c['thickness']) * (c['width'] - 2 * c['thickness'])
            * (c['height'] - c['thickness'] - c['buffer']))


def _catalog_item(rng, idx, jitter=0.0):
    base = dict(rng.choices(CATALOG, weights=CATALOG_WEIGHTS, k=1)[0])
    if jitter:
        for k in ('length', 'width', 'height'):
            base[k] = round(base[k] * rng.uniform(1 - jitter, 1 + jitter), 3)
    return _item(idx, base, False)


def _special_item(rng, idx, kind):
    if kind == 'small':
        d = dict(length=rng.uniform(0.25, 0.40), width=rng.uniform(0.15, 0.30),
                 height=rng.uniform(0.10, 0.20), mass=rng.uniform(2, 6), is_soft=rng.random() < 0.5)
    elif kind == 'golf':
        d = dict(length=rng.uniform(1.20, 1.35), width=rng.uniform(0.30, 0.40),
                 height=rng.uniform(0.30, 0.40), mass=rng.uniform(12, 20), is_soft=True)
    elif kind == 'stroller':
        d = dict(length=rng.uniform(0.90, 1.10), width=0.45, height=0.30,
                 mass=rng.uniform(7, 12), is_soft=False)
    elif kind == 'ski':
        d = dict(length=rng.uniform(1.60, 1.90), width=0.30, height=0.20,
                 mass=rng.uniform(8, 15), is_soft=True)
    elif kind == 'heavy':
        d = dict(length=0.75, width=0.56, height=0.30, mass=rng.uniform(25, 32), is_soft=False)
    else:
        raise ValueError(kind)
    for k in ('length', 'width', 'height', 'mass'):
        d[k] = round(d[k], 3)
    return _item(idx, d, False)


def _fill_items(rng, containers, supply, make):
    """Draw items until their volume reaches `supply` x container volume."""
    target = supply * sum(_inner_volume(c) for c in containers)
    items, vol = [], 0.0
    while vol < target and len(items) < 400:
        it = make(len(items))
        items.append(it)
        vol += it['length'] * it['width'] * it['height']
    return items


def _finish(cfg, items, k, max_space, prio_frac, rng, soft_frac=None):
    for i, it in enumerate(items):
        it['index'] = i
        if prio_frac:
            it['is_prioritized'] = rng.random() < prio_frac
        if soft_frac is not None:
            it['is_soft'] = rng.random() < soft_frac
            it.update(_item(i, dict(length=it['length'], width=it['width'], height=it['height'],
                                    mass=it['mass'], is_soft=it['is_soft']),
                            it['is_prioritized']))
    k = max(1, min(k, len(items)))
    cfg['item_stream']['item_list'] = items
    cfg['item_stream']['look_ahead'] = k
    cfg['item_stream']['max_space'] = max(1, min(max_space, k))
    cfg['item_stream']['visible_pool'] = items[:k]
    cfg['camera']['num_containers'] = len(cfg['containers']['container_list'])
    return cfg


def _prepack(rng, cfg, cidx, frac, where):
    """Settle-able pre-packed items: a floor grid of catalogue boxes at the
    back (realistic: transfer bags loaded first) or at the door (worst case)."""
    spacing = cfg['containers']['spacing']
    c = cfg['containers']['container_list'][cidx]
    L, W, t, buf = c['length'], c['width'], c['thickness'], c['buffer']
    target = frac * _inner_volume(c)
    xs0 = -L / 2 + t + c['cut_x'] + 0.03
    xs1 = L / 2 - t - 0.03
    placed, vol, z_base = [], 0.0, t + buf
    idx = 1000 + 100 * cidx
    layer = 0
    while vol < target and layer < 2:
        x = xs0
        yspan = (0.05, W / 2 - t - 0.03) if where == 'back' else (-W / 2 + 0.03, -0.05)
        while x < xs1 and vol < target:
            base = rng.choices(CATALOG, weights=CATALOG_WEIGHTS, k=1)[0]
            bl, bw, bh = base['length'], base['width'], base['height']
            if x + bl > xs1:
                break
            y = yspan[0]
            while y + bw <= yspan[1] and vol < target:
                it = _item(idx, base, rng.random() < 0.1)
                it['belongs_to'] = cidx
                it['pos'] = [round(cidx * spacing + x + bl / 2, 4), round(y + bw / 2, 4),
                             round(z_base + layer * (bh + 0.01) + bh / 2 + 0.005, 4)]
                it['orn'] = [0.0, 0.0, 0.0, 1.0]
                placed.append(it)
                idx += 1
                vol += bl * bw * bh
                y += bw + 0.03
            x += bl + 0.03
        layer += 1
    c['packed_items'] = placed


def build(seed=0):
    rng = random.Random(seed)
    out = {}
    n = 0

    def base(nc=1, shelf=False, prio=False, offline=False, k=10, geom=None):
        return make_scene(seed=rng.randrange(1 << 30), n_containers=nc, n_items=10,
                          shelf=shelf, prioritized=prio, look_ahead=k, offline=offline,
                          geom=geom or REAL_GEOMS[1 if shelf else 0])

    def cat(jit=0.0):
        return lambda i: _catalog_item(rng, i, jit)

    # buf: both shipped geometries x four buffers x (online k=10, offline k=1)
    for gi, shelf in ((0, False), (1, True)):
        for b in (0.005, 0.01, 0.02, 0.04):
            for offline, k in ((False, 10), (True, 1)):
                cfg = base(shelf=shelf, offline=offline, k=k)
                for c in cfg['containers']['container_list']:
                    c['buffer'] = b
                items = _fill_items(rng, cfg['containers']['container_list'], 0.65, cat())
                out[f'buf_{"S" if shelf else "N"}_b{int(b * 1000):02d}_{"A" if offline else "B"}'] = \
                    _finish(cfg, items, k, 1, 0.1, rng)

    # geo: random ULD-like sizes, with and without shelf, buffer 0 / 0.01
    for i in range(10):
        shelf = i % 2 == 1
        g = _geom(rng, shelf)
        cfg = base(shelf=shelf, offline=(i % 3 == 0), k=(1 if i % 3 == 0 else rng.choice([3, 10, 20])))
        for c in cfg['containers']['container_list']:
            _set_container(c, g, rng.choice([0.0, 0.01]), shelf, False)
        items = _fill_items(rng, cfg['containers']['container_list'], rng.uniform(0.6, 0.9), cat())
        out[f'geo_{i:02d}'] = _finish(cfg, items, cfg['item_stream']['look_ahead'], 1, 0.1, rng)

    # multi: 3-4 containers, mixed geometry, priority container, k 10-40
    for i in range(6):
        nc = 3 if i < 3 else 4
        k = (10, 20, 40)[i % 3]
        cfg = base(nc=nc, prio=True, k=k, offline=(i == 5))
        for j, c in enumerate(cfg['containers']['container_list']):
            shelf = j == 1
            g = REAL_GEOMS[1] if shelf else (REAL_GEOMS[0] if j != 2 else _geom(rng, False))
            _set_container(c, g, rng.choice([0.0, 0.01]), shelf, j == 0)
        items = _fill_items(rng, cfg['containers']['container_list'], 0.65, cat())
        out[f'multi_{nc}c_k{k:02d}_{i}'] = _finish(cfg, items, k, 1, 0.15, rng)

    # pre: pre-packed 10-40% at the back floor or the door side
    for i, (frac, where) in enumerate([(0.10, 'back'), (0.25, 'back'), (0.40, 'back'),
                                       (0.10, 'door'), (0.25, 'door'), (0.15, 'back')]):
        shelf = i % 2 == 1
        cfg = base(shelf=shelf, k=rng.choice([3, 10]), offline=(i == 5))
        _prepack(rng, cfg, 0, frac, where)
        items = _fill_items(rng, cfg['containers']['container_list'], 0.65 - frac, cat())
        out[f'pre_{where}_{int(frac * 100):02d}_{i}'] = _finish(
            cfg, items, cfg['item_stream']['look_ahead'], 1, 0.1, rng)

    # item: jittered catalogue plus special items
    item_mixes = [
        ('small30', dict(small=0.30)), ('small50', dict(small=0.50)),
        ('long', dict(golf=0.06, stroller=0.04, ski=0.03)),
        ('heavy', dict(heavy=0.15)), ('mixed', dict(small=0.2, golf=0.04, heavy=0.05)),
    ]
    for i, (name, mix) in enumerate(item_mixes):
        for offline, k in ((False, 10), (True, 1)):
            shelf = (i + offline) % 2 == 1
            cfg = base(shelf=shelf, offline=offline, k=k)

            def make(idx, _mix=mix):
                r = rng.random()
                acc = 0.0
                for kind, p in _mix.items():
                    acc += p
                    if r < acc:
                        return _special_item(rng, idx, kind)
                return _catalog_item(rng, idx, 0.2)

            items = _fill_items(rng, cfg['containers']['container_list'], 0.7, make)
            out[f'item_{name}_{"A" if offline else "B"}'] = _finish(
                cfg, items, k, 1, rng.uniform(0.0, 0.3), rng, soft_frac=rng.uniform(0.1, 0.8))

    # flow: look_ahead x max_space, ordering patterns
    for k in (3, 10, 20, 40):
        for ms in sorted({1, max(1, k // 2), k}):
            if ms == 1:
                continue            # max_space=1 is what every local pool already has
            cfg = base(shelf=(k in (10, 40)), k=k)
            items = _fill_items(rng, cfg['containers']['container_list'], 0.65, cat())
            out[f'flow_k{k:02d}_ms{ms:02d}'] = _finish(cfg, items, k, ms, 0.1, rng)
    for pattern in ('bigLate', 'prioFirst', 'prioLast', 'softFirst'):
        cfg = base(shelf=pattern in ('bigLate', 'prioLast'), k=5)
        items = _fill_items(rng, cfg['containers']['container_list'], 0.7, cat())
        vol = lambda it: it['length'] * it['width'] * it['height']
        if pattern == 'bigLate':
            items.sort(key=vol)
        elif pattern == 'softFirst':
            items.sort(key=lambda it: not it['is_soft'])
        cfg = _finish(cfg, items, 5, 1, 0.0, rng)
        if pattern.startswith('prio'):
            n_p = max(3, len(items) // 5)
            sel = range(n_p) if pattern == 'prioFirst' else range(len(items) - n_p, len(items))
            for j in sel:
                items[j]['is_prioritized'] = True
        out[f'flow_{pattern}'] = cfg

    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='scenes_stress.json')
    a = ap.parse_args()
    scenes = build(a.seed)
    json.dump(scenes, open(a.out, 'w'))
    print('wrote', len(scenes), 'scenes to', a.out)
