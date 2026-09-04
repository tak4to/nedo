import json, random
from scenegen import make_scene
rng = random.Random(77)
pool = {}
for i in range(16):
    nc = rng.choice([1,1,2])
    pool[f'O{i:02d}'] = make_scene(
        seed=5000+i, n_containers=nc, n_items=rng.choice([40,45,50])*nc,
        shelf=rng.random()<0.35, prioritized=(nc==2 and rng.random()<0.4),
        look_ahead=1, offline=True, prio_frac=rng.choice([0.0,0.1,0.2]))
json.dump(pool, open('scenes_offline.json','w'))
print('wrote', len(pool))
