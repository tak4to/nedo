"""Scene generator: builds config dicts like sample_config.json."""
import json, random, copy

CATALOG = [
    dict(length=0.75, width=0.56, height=0.27, mass=18, is_soft=False),
    dict(length=0.65, width=0.45, height=0.25, mass=13, is_soft=False),
    dict(length=0.55, width=0.40, height=0.24, mass=8,  is_soft=False),
    dict(length=0.60, width=0.30, height=0.25, mass=7,  is_soft=True),
    dict(length=0.50, width=0.40, height=0.40, mass=10, is_soft=True),
    dict(length=0.65, width=0.35, height=0.23, mass=12, is_soft=True),
    dict(length=0.45, width=0.30, height=0.20, mass=5,  is_soft=True),
]

# Empirical frequencies of the seven types across both shipped tasks
# (configs/sample_config.json, 83 items). The generator used to sample the
# catalogue uniformly, which produced 57% soft items against the real 29%
# and far more of the largest type than the real streams contain -- a
# materially different packing problem from the one being scored.
CATALOG_WEIGHTS = [7.2, 30.1, 33.7, 6.0, 4.8, 14.5, 3.6]


def _item(idx, base, prio):
    d = dict(base); d['index'] = idx
    d['is_prioritized'] = prio
    d['lateralFriction'] = 0.8 if d['is_soft'] else 0.4
    d['rollingFriction'] = 0.02 if d['is_soft'] else 0.01
    d['spinningFriction'] = 0.02 if d['is_soft'] else 0.01
    d['restitution'] = 0.1 if d['is_soft'] else 0.2
    if d['is_soft']:
        d['contactStiffness'] = 2500; d['contactDamping'] = 700; d['linearDamping'] = 0.8
    return d

# The two container geometries the competition actually ships
# (configs/sample_config.json). They are NOT interchangeable: the shelf
# task is the wider, thicker one, and until 2026-09-03 this generator
# built every scene -- shelf scenes included -- with task 000's geometry,
# so no shelf scene was ever measured against the real shelf container.
REAL_GEOMS = [
    dict(L=2.0, W=1.45, H=1.61, thickness=0.04, cut_x=0.44, cut_y=0.4),   # task 000
    dict(L=2.0, W=1.52, H=1.62, thickness=0.05, cut_x=0.43, cut_y=0.4),   # task 001 (shelf)
]


def make_container(index, prioritized=False, shelf=False, L=2.0, W=1.45, H=1.61,
                   thickness=0.04, cut_x=0.44, cut_y=0.4):
    return dict(index=index, length=L, width=W, height=H, thickness=thickness, buffer=0.0,
                cut_x=cut_x, cut_y=cut_y, packed_items=[], require_shelf=shelf,
                is_prioritized=prioritized)

def make_scene(seed, n_containers=1, n_items=45, shelf=False, prioritized=False,
               look_ahead=1, offline=False, prio_frac=0.1, L=2.0, W=1.45, H=1.61,
               geom=None):
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        base = rng.choices(CATALOG, weights=CATALOG_WEIGHTS, k=1)[0]
        items.append(_item(i, base, rng.random() < prio_frac))
    containers = []
    gkw = dict(L=L, W=W, H=H) if geom is None else dict(geom)
    for c in range(n_containers):
        containers.append(make_container(c, prioritized=(prioritized and c == 0),
                                         shelf=(shelf and c == 0), **gkw))
    k = min(look_ahead, n_items)
    # max_space is the refill trigger, NOT the pool size: ItemStreamManager
    # refills only once (lookahead_k - len(pool)) >= max_space. The real
    # configs ship max_space=1 (configs/sample_config.json, both tasks), i.e.
    # a true sliding window that tops the pool back up after every single
    # placement. Setting it to k instead -- which this generator did until
    # 2026-08-30 -- makes the pool drain to empty and refill in batches, a
    # materially different problem. Anything measured on look_ahead>1 scenes
    # before that fix was measuring the wrong dynamics.
    cfg = {
        'containers': {'spacing': 2.5, 'container_list': containers},
        'item_stream': {'item_list': items, 'look_ahead': k, 'max_space': 1,
                        'visible_pool': items[:k]},
        'camera': {'num_containers': max(n_containers, 1), 'target_pos': [0, 0, 0],
                   'distance': 3.0, 'yaw': 0, 'pitch': 0, 'roll': 0,
                   'img_width': 64, 'img_height': 64, 'fov': 60,
                   'near_val': 0.1, 'far_val': 10.0},
        'validator': {'inclusion_margin': -0.005, 'start_z': 0.08, 'safety_margin': 0.015,
                      'ceiling_margin': 0.018, 'displacement_threshold': 0.3,
                      'angle_displacement_threshold': 45, 'settle_wait_step': 300},
        'action': {'keys': {'item_idx': 'int', 'container_idx': 'int',
                            'place_pos': 'float', 'orientation': 'int'},
                   'pos_lim': {'low': -100, 'high': 100},
                   'orientations': [0, 1, 2, 3, 4, 5]},
        'agent': {'optimize': offline, 'init_timeout': 10.0, 'optimization_timeout': 180.0,
                  'policy_timeout': 8.0,
                  'allowed_methods': ['get_init_states', 'optimize', 'policy'], 'max_mem': 12},
        'visualizer': {'vis': False, 'camera': {'yaw': 0, 'pitch': -20}},
    }
    return cfg

SCENES = {
  'A_1c_offline':   dict(seed=1, n_containers=1, n_items=45, offline=True,  look_ahead=1),
  'B_shelf_pool10': dict(seed=2, n_containers=1, n_items=45, shelf=True,    look_ahead=10),
  'C_lookahead1':   dict(seed=3, n_containers=1, n_items=45, look_ahead=1),
  'D_2c_offline':   dict(seed=4, n_containers=2, n_items=80, offline=True,  look_ahead=1),
  'E_2c_priority':  dict(seed=5, n_containers=2, n_items=80, prioritized=True, look_ahead=5, prio_frac=0.2),
  'F_2c_shelf':     dict(seed=6, n_containers=2, n_items=80, shelf=True,    look_ahead=3),
  'G_pool40':       dict(seed=7, n_containers=1, n_items=45, look_ahead=40),
}

if __name__ == '__main__':
    out = {name: make_scene(**kw) for name, kw in SCENES.items()}
    with open('scenes.json', 'w') as f:
        json.dump(out, f)
    print('wrote', len(out), 'scenes')
