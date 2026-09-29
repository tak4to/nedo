"""Scene pools at the density the real operation runs at: 50-60 bags per
container.

JAL's Haneda ground-handling demo puts 50-60 checked bags in one container
(docs/2026-09-23-現場の実務と再現・優先度.md §1); every pool built so far
uses 40-50 per container, so none of them has ever shown how the solver
behaves once the stream carries clearly more bags than the container can
take. That is the regime the operation is actually in -- the container
fills and the rest goes to the next one -- and it is where "how far does it
get before it jams" is decided.

Same structure as poolgen.py (mixed task types, shelf/priority variants,
real container geometries); only the item count per container changes.
Dev and holdout on disjoint seeds, and disjoint from every other pool.

    cd tools && ../.venv/bin/python poolgen_dense.py
"""
import json
import random

from scenegen import REAL_GEOMS, make_scene


def build(seed0, tag, n=32):
    rng = random.Random(seed0)
    pool = {}
    for i in range(n):
        nc = rng.choice([1, 1, 2])
        shelf = rng.random() < 0.35
        pool[f'{tag}{i:02d}'] = make_scene(
            seed=seed0 + 11 * i, n_containers=nc,
            n_items=rng.choice([50, 55, 60]) * nc,
            shelf=shelf,
            geom=REAL_GEOMS[1] if shelf else REAL_GEOMS[0],
            prioritized=(nc == 2 and rng.random() < 0.4),
            look_ahead=rng.choice([1, 1, 3, 5, 10, 20]),
            offline=rng.random() < 0.4,
            prio_frac=rng.choice([0.0, 0.1, 0.2]))
    return pool


if __name__ == '__main__':
    json.dump(build(41000, 'D'), open('scenes_dense.json', 'w'))
    json.dump(build(83000, 'E'), open('scenes_densetest.json', 'w'))
    print('wrote scenes_dense.json / scenes_densetest.json (50-60 items per container)')
