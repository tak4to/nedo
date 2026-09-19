"""Offline-mode pools: optimize=True paired with look_ahead=1.

The real configs couple the two -- task 000 is (optimize=True, k=1) and task
001 is (optimize=False, k=10) -- but poolgen.py samples `offline` and
`look_ahead` independently, so the existing pools hold exactly one
(True, k=1) scene between them (scenes_pool P**) and several (True, k=20)
and (True, k=40) scenes that the real distribution does not contain.

That is not a cosmetic mismatch. `optimize`'s rollout sets
`allowed_misses = max(1, lookahead_k)`, so at k=1 it stops at the first
placement it cannot make -- what the real episode does -- while at k=20 it
carries on past twenty of them. Measuring the offline search on high-k
scenes measures a different algorithm.

This is the fourth time the scene generator's distribution has been the bug
(see docs/2026-09-03-課題分析.md for the container-dimension and
item-frequency ones), so: real geometries, real catalogue weights, and the
real (optimize, look_ahead) pairing.
"""
import json
import random

from scenegen import REAL_GEOMS, make_scene


def build(seed0, tag, n=64):
    rng = random.Random(seed0)
    pool = {}
    for i in range(n):
        nc = rng.choice([1, 1, 2])
        shelf = rng.random() < 0.35
        pool[f'{tag}{i:02d}'] = make_scene(
            seed=seed0 + 13 * i, n_containers=nc,
            n_items=rng.choice([40, 45, 50]) * nc,
            shelf=shelf,
            # The one real shelf task ships the wider, thicker container, so
            # keep shelf and geometry correlated the way the configs do.
            geom=REAL_GEOMS[1] if shelf else REAL_GEOMS[0],
            prioritized=(nc == 2 and rng.random() < 0.4),
            look_ahead=1, offline=True,
            prio_frac=rng.choice([0.0, 0.1, 0.2]))
    return pool


if __name__ == '__main__':
    json.dump(build(41000, 'O'), open('scenes_off.json', 'w'))
    json.dump(build(83000, 'Q'), open('scenes_offtest.json', 'w'))
    print('wrote scenes_off.json / scenes_offtest.json (64 each, optimize=True k=1)')
