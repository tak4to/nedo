"""Where do placement / soft violations come from?

Runs scenes with an agent and records, for every placed item, which part of
the policy chose it: 'primary' (the normal search), 'rescue' (the fallback
ladder found something) or 'last_resort' (the unvalidated final move). At
the end every soft / priority item is checked for being buried by an item
of another class under two definitions:

  column   an item of another class anywhere above it with overlapping
           footprint (what tools/scores.py placement_scores counts)
  contact  an item of another class physically touching it from above
           (PyBullet contact points; README: 上方向からの接触判定がある)

and each violating pair is attributed to the source of the upper item.

    AGENT_DIR=../agents/hf3 python viol_trace.py --scenes a.json,b.json [--jobs 14] --tag T
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = '/home/takato/comp/nedo'


def _boxes(env):
    out = []
    for c in env.container_manager.containers:
        for it in c.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            R = np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            h = np.abs(R) @ np.array([it.length / 2, it.width / 2, it.height / 2])
            out.append(dict(it=it, c=c, pos=np.array(pos), h=h))
    return out


def violations(env):
    bs = _boxes(env)
    out = []
    for kind in ('soft', 'prio'):
        cls = (lambda it: it.is_soft) if kind == 'soft' else (lambda it: it.is_prioritized)
        for b in bs:
            if not cls(b['it']):
                continue
            top = b['pos'][2] + b['h'][2]
            for o in bs:
                if o is b or o['c'] is not b['c'] or cls(o['it']) == cls(b['it']):
                    continue
                if o['pos'][2] - o['h'][2] < top - 0.02:
                    continue
                if not (abs(o['pos'][0] - b['pos'][0]) < o['h'][0] + b['h'][0] and
                        abs(o['pos'][1] - b['pos'][1]) < o['h'][1] + b['h'][1]):
                    continue
                gap = (o['pos'][2] - o['h'][2]) - top
                cps = env.client.getContactPoints(bodyA=b['it'].pybullet_id, bodyB=o['it'].pybullet_id)
                touching = any(cp[8] < 0.005 for cp in cps)
                out.append(dict(kind=kind, lower=b['it'].index, upper=o['it'].index,
                                gap=round(float(gap), 4), contact=bool(touching)))
    return out


def _one(args):
    name, cfg, agent_dir, pol, opt = args
    sys.path.insert(0, ROOT)
    sys.path.insert(0, agent_dir)
    import agent as agent_module
    importlib.reload(agent_module)
    import packer
    from src.ground_handling.env import GroundHandlingEnv
    agent_module.POLICY_TIME_BUDGET = pol
    agent_module.OPTIMIZE_TIME_BUDGET = opt
    flag = {}
    orig_fb = agent_module._fallback_action

    def fb(*a, **k):
        flag['fallback'] = True
        return orig_fb(*a, **k)
    agent_module._fallback_action = fb
    if hasattr(packer, 'rescue_action'):
        orig_ra = packer.rescue_action

        def ra(*a, **k):
            r = orig_ra(*a, **k)
            flag['rescue'] = r is not None
            return r
        packer.rescue_action = ra
    source = {}
    with contextlib.redirect_stdout(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = agent_module.Agent(module_path='')
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        term, steps = False, 0
        while not term and steps < 500:
            flag.clear()
            before = {id(it) for c in env.container_manager.containers for it in c.packed_items}
            a = ag.policy(observation=obs)
            if not flag.get('fallback'):
                src = 'primary'
            elif flag.get('rescue', True):
                src = 'rescue'
            else:
                src = 'last_resort'
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            for c in env.container_manager.containers:
                for it in c.packed_items:
                    if id(it) not in before:
                        source[it.index] = src
        v = violations(env)
        n_soft = sum(1 for c in env.container_manager.containers for it in c.packed_items if it.is_soft)
        n_prio = sum(1 for c in env.container_manager.containers for it in c.packed_items if it.is_prioritized)
        packed = sum(len(c.packed_items) for c in env.container_manager.containers)
        env.close()
    for x in v:
        x['upper_src'] = source.get(x['upper'], 'prepacked')
    return dict(scene=name, packed=packed, n_soft=n_soft, n_prio=n_prio,
                src_counts=dict(Counter(source.values())), viol=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--policy-budget', type=float, default=5.0)
    ap.add_argument('--optimize-budget', type=float, default=60.0)
    ap.add_argument('--tag', default='vt')
    a = ap.parse_args()
    agent_dir = os.environ.get('AGENT_DIR') or os.path.join(ROOT, 'agents', 'submit')
    scenes = {}
    for p in a.scenes.split(','):
        pre = os.path.splitext(os.path.basename(p))[0].replace('scenes_', '')
        scenes.update({f'{pre}:{k}': v for k, v in json.load(open(p)).items()})
    jobs = [(n, c, agent_dir, a.policy_budget, a.optimize_budget) for n, c in scenes.items()]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_one, jobs))
    json.dump(rows, open(f'viol_{a.tag}.json', 'w'), indent=1)
    report(rows)


def report(rows):
    V = [x for r in rows for x in r['viol']]
    n_soft = sum(r['n_soft'] for r in rows)
    n_prio = sum(r['n_prio'] for r in rows)
    src = Counter()
    for r in rows:
        src.update(r['src_counts'])
    print(f'{len(rows)} scenes, placed items by source: {dict(src)}; soft items {n_soft}, priority {n_prio}')
    for kind in ('soft', 'prio'):
        vk = [x for x in V if x['kind'] == kind]
        lower_col = {(r['scene'], x['lower']) for r in rows for x in r['viol'] if x['kind'] == kind}
        lower_con = {(r['scene'], x['lower']) for r in rows for x in r['viol'] if x['kind'] == kind and x['contact']}
        print(f'  {kind}: buried items  column-rule {len(lower_col)}  contact-rule {len(lower_con)}')
        for rule, sel in (('column', vk), ('contact', [x for x in vk if x['contact']])):
            c = Counter(x['upper_src'] for x in sel)
            print(f'    {rule:7s} pairs {len(sel):4d} by upper item source: {dict(c)}')
        gaps = sorted(x['gap'] for x in vk)
        if gaps:
            print(f'    vertical gap between them (column pairs): median {gaps[len(gaps) // 2]:.3f} m, '
                  f'<=0.03 m: {sum(1 for g in gaps if g <= 0.03)}')


if __name__ == '__main__':
    main()
