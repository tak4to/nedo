"""Is 'how many (type,orientation) classes still have a feasible slot' a
usable early-warning signal for the absorbing failure?"""
import os,sys,json,io,contextlib,time,math
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from src.ground_handling.env import GroundHandlingEnv
import geometry, packer, agent as agent_module
from scenegen import CATALOG

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def alive_vector(cstates, cap=4000):
    """For each catalog type: does ANY container x orientation admit it?
    Returns (n_alive_types, n_alive_classes, cost_in_validations)."""
    n_types=0; n_classes=0
    t0=time.perf_counter()
    for base in CATALOG:
        spec=dict(base); spec['index']=-1; spec['is_prioritized']=False
        type_alive=False
        for cs in cstates.values():
            r=packer.best_placement(cs, spec, time.perf_counter()+0.25)
            if r is not None:
                type_alive=True; n_classes+=1
                break
        if type_alive: n_types+=1
    return n_types, n_classes, time.perf_counter()-t0

def run(name,cfg,pol=5.0,opt=60.0):
    agent_module.POLICY_TIME_BUDGET=pol; agent_module.OPTIMIZE_TIME_BUDGET=opt
    trace=[]
    with quiet():
        env=GroundHandlingEnv(config=cfg,verbose=False)
        ag=agent_module.Agent(module_path='')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs,info=env.reset(seed=42)
        term=False; step=0; last={}
        while not term and step<500:
            cl=obs.get('container_list') or []
            cstates={c['index']:packer.ContainerState(c) for c in cl}
            nt,nc,cost=alive_vector(cstates)
            trace.append((step,nt,nc,cost))
            a=ag.policy(observation=obs)
            obs,r,term,trunc,last=env.step(a)
            step+=1
    return trace,last,step

if __name__=='__main__':
    scenes=json.load(open('scenes_pool.json'))
    names=sys.argv[1:] or ['P05','P13','P17']
    for nm in names:
        cfg=scenes[nm]
        tr,last,steps=run(nm,cfg)
        ncont=len(cfg['containers']['container_list'])
        print(f"\n=== {nm} ({ncont} container(s)) died at step {steps}, status={last.get('status')}")
        print(f"{'step':>4s} {'alive_types/7':>13s} {'probe_s':>8s}")
        for (s,nt,nc,cost) in tr:
            bar='#'*nt
            print(f"{s:4d} {nt:6d}/7      {cost:7.3f}  {bar}")
