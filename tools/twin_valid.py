"""Does the twin predict the real settle failures?

Replays each scene, remembers the observation before every step, and when
the episode ends on is_placed_safe=False rebuilds a twin from that
observation and asks it whether the fatal action settles.
"""
import os, sys, json, math, importlib, io, contextlib
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.environ.get('AGENT_DIR',ROOT+'/agents/submit')); sys.path.insert(0,'.')
import pybullet as p
from src.ground_handling.env import GroundHandlingEnv
from src.ground_handling.utils import ORNS
from twin_bench import Twin

def one(name,cfg,steps_n):
    import agent as am; importlib.reload(am)
    am.POLICY_TIME_BUDGET=5.0; am.OPTIMIZE_TIME_BUDGET=60.0
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False,render_mode=None)
        ag=am.Agent(module_path=ROOT+'/agents/submit')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream(); obs,info=env.reset(seed=42)
        term=False; n=0; prev=None; act=None
        while not term and n<500:
            act=ag.policy(observation=obs); prev=obs
            obs,r,term,tr,info=env.step(act); n+=1
        st=info.get('status') or {}
        packed=sum(len(c.packed_items) for c in env.container_manager.containers)
        total=env.num_total_items
        res=None
        if st.get('is_valid') and not st.get('is_placed_safe'):
            cd=[c for c in prev['container_list'] if c['index']==int(act['container_idx'])][0]
            spec=prev['pool_list'][int(act['item_idx'])]
            import geometry as _g
            tw=Twin(cd,_g.container_geometry(cd)); tw.sync(cd['packed_items'])
            q=p.getQuaternionFromEuler(ORNS[int(act['orientation'])])
            pos=(float(act['place_pos'][0]),float(act['place_pos'][1]),float(act['place_pos'][2]))
            d,a=tw.settles(spec,pos,q,steps_n)
            res=(d,a)
            tw.cl.disconnect()
        env.close()
    return st,packed,total,res

if __name__=='__main__':
    steps_n=int(sys.argv[1]) if len(sys.argv)>1 else 120
    scenes=json.load(open('scenes_pool.json'))
    if len(sys.argv)>2:
        keep=set(sys.argv[2:]); scenes={k:v for k,v in scenes.items() if k in keep}
    hit=miss=0
    for nm,cfg in scenes.items():
        st,pk,tot,res=one(nm,cfg,steps_n)
        if res is None: continue
        d,a=res
        flagged = d>0.3 or a>45
        hit+= flagged; miss+= (not flagged)
        print(f'{nm}: topple  twin disp={d:.3f} angle={a:.1f} -> {"CAUGHT" if flagged else "missed"}  ({pk}/{tot})',flush=True)
    print(f'\ntwin caught {hit} of {hit+miss} real topples (settle steps={steps_n})')
