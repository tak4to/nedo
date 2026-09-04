import json, random
from scenegen import make_scene
rng = random.Random(20260830)
pool = {}
for i in range(32):
    nc = rng.choice([1,1,2])
    pool[f'P{i:02d}'] = make_scene(
        seed=1000+i, n_containers=nc,
        n_items=rng.choice([40,45,50]) * nc,
        shelf=rng.random() < 0.35,
        prioritized=(nc == 2 and rng.random() < 0.4),
        look_ahead=rng.choice([1,1,3,5,10,20]),
        offline=rng.random() < 0.4,
        prio_frac=rng.choice([0.0,0.1,0.2]))
json.dump(pool, open('scenes_pool.json','w'))
print('wrote', len(pool))
