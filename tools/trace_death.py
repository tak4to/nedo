"""Step-by-step trace of one scene: what was placed, and what the validator said
about the last (terminating) step. Local diagnostic only."""
import os, sys, json, io, contextlib, time
ROOT = '/home/takato/comp/nedo'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'agents', 'submit'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import numpy as np
from src.ground_handling.env import GroundHandlingEnv
import agent as agent_module

def trace(name, cfg, scenes_file, opt_budget=15.0, last=4):
    agent_module.OPTIMIZE_TIME_BUDGET = opt_budget
    log = []
    with contextlib.redirect_stdout(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = agent_module.Agent(module_path=os.path.join(ROOT, 'agents', 'submit'))
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, _ = env.reset(seed=42)
        term = False; steps = 0
        while not term and steps < 500:
            pool = obs['pool_list']
            action = ag.policy(observation=obs)
            it = pool[action['item_idx']] if action['item_idx'] < len(pool) else None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                obs, r, term, trunc, info = env.step(action)
            log.append((steps, it, action, info.get('status'), buf.getvalue().strip()))
            steps += 1
    print(f'== {name}: {steps} steps')
    for s, it, a, st, msg in log[-last:]:
        d = (it['length'], it['width'], it['height'], it['mass']) if it else None
        print(f" step {s}: item {it and it['index']} dims/mass={d} soft={it and it.get('is_soft')} prio={it and it.get('is_prioritized')}"
              f" cont={a['container_idx']} orn={a['orientation']} pos={np.round(a['place_pos'],3).tolist()}\n    status={st} | {msg[-200:]}")
    return log

if __name__ == '__main__':
    f = sys.argv[1]; names = sys.argv[2:]
    sc = json.load(open(f))
    for n in names:
        trace(n, sc[n], f)

def layout(name, cfg, opt_budget=15.0):
    """Final layout (before the terminating item) per container."""
    agent_module.OPTIMIZE_TIME_BUDGET = opt_budget
    with contextlib.redirect_stdout(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = agent_module.Agent(module_path=os.path.join(ROOT, 'agents', 'submit'))
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, _ = env.reset(seed=42)
        term = False; steps = 0
        while not term and steps < 500:
            obs, r, term, trunc, info = env.step(ag.policy(observation=obs)); steps += 1
        out = []
        for ci, c in enumerate(env.container_manager.containers):
            for it in c.packed_items:
                pos, orn = it.get_pose(env.client)
                out.append((ci, it.index, np.round(pos, 2).tolist(), (it.length, it.width, it.height), it.is_soft, it.is_prioritized))
        env.close()
    return out
