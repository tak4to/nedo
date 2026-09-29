"""Terminal fork: how many items does the end-game time budget cost?

For each scene: run the agent normally and record every action (and the
offline order, if any). Replay all actions but the last into a fresh env --
the state right before the move that ended the episode -- and ask the agent
again with an unlimited policy budget, so the primary search and every rung
of the rescue ladder run to completion. If that finds a placement the env
accepts, keep going with the normal budget and count what else gets placed.

The result is an upper bound on what any re-allocation of the policy's time
at the end game can recover with the current candidate generator (plan v3,
C-1). It says nothing about placements the generator never proposes.

    AGENT_DIR=../agents/submit python term_fork.py --scenes s.json [--only A,B] --jobs 12 --tag T

Candidate mode: replay the recorded runs of an earlier tag into the same
pre-terminal states and let ANOTHER agent make that decision on its normal
budget -- how much of the upper bound a real change recovers, measured on
identical states:

    AGENT_DIR=../agents/hf3 python term_fork.py --scenes s.json --replay-from T --big 5.0 --tag T2
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/takato/comp/nedo'


def _env(cfg):
    from src.ground_handling.env import GroundHandlingEnv
    return GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)


def _status(info):
    st = (info or {}).get('status') or {}
    if not st.get('is_valid', True):
        return 'stuck'
    if not st.get('is_placed_safe', True):
        return 'topple'
    if not st.get('is_included', True):
        return 'outside'
    return 'ok'


def _act(a):
    return {'item_idx': int(a['item_idx']), 'container_idx': int(a['container_idx']),
            'place_pos': [float(v) for v in a['place_pos']], 'orientation': int(a['orientation'])}


def _np_act(a):
    import numpy as np
    return dict(a, place_pos=np.array(a['place_pos'], dtype=np.float32))


def _one(args):
    name, cfg, agent_dir, pol, opt, big, rec = args
    sys.path.insert(0, ROOT)
    sys.path.insert(0, agent_dir)
    import agent as agent_module
    importlib.reload(agent_module)
    agent_module.POLICY_TIME_BUDGET = pol
    agent_module.OPTIMIZE_TIME_BUDGET = opt
    out = dict(scene=name)
    with contextlib.redirect_stdout(io.StringIO()):
        if rec is not None:
            # candidate mode: the recorded run stands in for step 1
            actions = [_np_act(a) for a in rec['actions']]
            order = rec['order']
            for k in ('packed', 'total', 'end'):
                out[k] = rec[k]
        else:
            # 1. normal run, recording actions
            env = _env(cfg)
            ag = agent_module.Agent(module_path='')
            env.reset_settings()
            ag.get_init_states(env.get_init_states())
            order = None
            if cfg['agent']['optimize']:
                order = ag.optimize(env.get_info_for_optimization())
                env.set_item_order(order)
            env.reset_item_stream()
            obs, info = env.reset(seed=42)
            actions, term, info = [], False, {}
            while not term and len(actions) < 500:
                a = ag.policy(observation=obs)
                actions.append(a)
                obs, r, term, trunc, info = env.step(a)
            out['packed'] = sum(len(c.packed_items) for c in env.container_manager.containers)
            out['total'] = env.num_total_items
            out['end'] = _status(info) if out['packed'] < out['total'] else 'all'
            env.close()
        out['actions'] = [_act(a) for a in actions]
        out['order'] = [int(i) for i in order] if order is not None else None
        if out['end'] in ('all', 'ok'):
            out['rescued'] = False
            out['extra'] = 0
            return out

        # 2. replay to the state before the terminal move
        env = _env(cfg)
        ag = agent_module.Agent(module_path='')
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if order is not None:
            env.set_item_order(order)
        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        term = False
        for a in actions[:-1]:
            obs, r, term, trunc, info = env.step(a)
            if term:
                break
        replay_packed = sum(len(c.packed_items) for c in env.container_manager.containers)
        out['replay_ok'] = (not term) and replay_packed == out['packed']

        # 3. the same decision: unlimited budget (upper bound) or, in
        #    candidate mode, the candidate agent's own budget via --big
        agent_module.POLICY_TIME_BUDGET = big
        t0 = time.perf_counter()
        a = ag.policy(observation=obs)
        out['fork_s'] = time.perf_counter() - t0
        agent_module.POLICY_TIME_BUDGET = pol
        obs, r, term, trunc, info = env.step(a)
        out['rescued'] = not term
        out['fork_end'] = _status(info) if term else None
        # 4. continue normally
        while not term:
            a = ag.policy(observation=obs)
            obs, r, term, trunc, info = env.step(a)
        fork_packed = sum(len(c.packed_items) for c in env.container_manager.containers)
        out['fork_packed'] = fork_packed
        out['extra'] = fork_packed - out['packed']
        env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True, help='comma-separated scene files')
    ap.add_argument('--only', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--policy-budget', type=float, default=5.0)
    ap.add_argument('--optimize-budget', type=float, default=60.0)
    ap.add_argument('--big', type=float, default=600.0)
    ap.add_argument('--tag', default='fork')
    ap.add_argument('--replay-from', default=None, help='tag of an earlier run to replay')
    a = ap.parse_args()
    agent_dir = os.environ.get('AGENT_DIR') or os.path.join(ROOT, 'agents', 'submit')
    scenes = {}
    for p in a.scenes.split(','):
        pre = os.path.splitext(os.path.basename(p))[0].replace('scenes_', '')
        scenes.update({f'{pre}:{k}': v for k, v in json.load(open(p)).items()})
    if a.only:
        keep = a.only.split(',')
        scenes = {k: v for k, v in scenes.items() if any(k.split(':', 1)[1].startswith(x) or k.startswith(x) for x in keep)}
    recs = {}
    if a.replay_from:
        recs = {r['scene']: r for r in json.load(open(f'fork_{a.replay_from}.json'))}
        scenes = {k: v for k, v in scenes.items() if k in recs}
    jobs = [(n, c, agent_dir, a.policy_budget, a.optimize_budget, a.big, recs.get(n))
            for n, c in scenes.items()]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_one, jobs))
    rows.sort(key=lambda r: r['scene'])
    json.dump(rows, open(f'fork_{a.tag}.json', 'w'), indent=1)
    stuck = [r for r in rows if r['end'] not in ('all', 'ok')]
    resc = [r for r in stuck if r.get('rescued')]
    for r in rows:
        print(f"  {r['scene']:28s} {r['packed']:3d}/{r['total']:3d} end={r['end']:7s} "
              f"rescued={r.get('rescued')!s:5s} extra={r.get('extra', 0):+d} "
              f"fork_s={r.get('fork_s', 0):6.1f} replay_ok={r.get('replay_ok')}")
    by_end = {}
    for r in stuck:
        by_end.setdefault(r['end'], []).append(r)
    print(f'== {a.tag}: {len(rows)} scenes, {len(stuck)} ended early, rescued {len(resc)}; '
          f'extra items per early-ended scene {sum(r["extra"] for r in stuck) / max(len(stuck), 1):.2f}')
    for e, v in by_end.items():
        print(f'   end={e:7s} n={len(v):3d} rescued={sum(1 for r in v if r.get("rescued")):3d} '
              f'extra/scene={sum(r["extra"] for r in v) / len(v):.2f}')


if __name__ == '__main__':
    main()
