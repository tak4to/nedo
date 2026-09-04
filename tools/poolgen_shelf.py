"""Shelf-only pools. Shelf containers are the weakest segment measured
(50.7% of items placed vs 63.7% without a shelf, over 64 mixed scenes),
they are 18/64 of the mixed pools -- too few to diagnose against -- and
the real sample task 001 has require_shelf=True."""
import json, random
from scenegen import make_scene, REAL_GEOMS

def build(seed0, tag, n=32):
    rng = random.Random(seed0)
    pool = {}
    for i in range(n):
        nc = rng.choice([1, 1, 2])
        pool[f'{tag}{i:02d}'] = make_scene(
            seed=seed0 + 13 * i, n_containers=nc,
            n_items=rng.choice([40, 45, 50]) * nc,
            shelf=True,
            geom=REAL_GEOMS[1] if rng.random() < 0.7 else REAL_GEOMS[0],
            prioritized=(nc == 2 and rng.random() < 0.4),
            look_ahead=rng.choice([1, 3, 5, 10, 10, 20]),
            offline=rng.random() < 0.4,
            prio_frac=rng.choice([0.0, 0.1, 0.2]))
    return pool

if __name__ == '__main__':
    json.dump(build(31000, 'S'), open('scenes_shelf.json', 'w'))
    json.dump(build(77000, 'T'), open('scenes_shelftest.json', 'w'))
    print('wrote scenes_shelf.json / scenes_shelftest.json (real geometries)')
