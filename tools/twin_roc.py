"""TPR/FPR of the twin at a reduced step count, over EVERY step of an episode."""
import os, sys, json, math, importlib, io, contextlib, time
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.environ.get('AGENT_DIR',ROOT+'/agents/submit')); sys.path.insert(0,'.')
import pybullet as p
from src.ground_handling.env import GroundHandlingEnv
from src.ground_handling.utils import ORNS
from twin_bench import Twin
import geometry

def one(name,cfg,steps_n):
    import agent as am; importlib.reload(am)
    am.POLICY_TIME_BUDGET=5.0; am.OPTIMIZE_TIME_BUDGET=60.0
    rows=[]; tsum=0.0
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False,render_mode=None)
        ag=am.Agent(module_path=ROOT+'/agents/submit')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream(); obs,info=env.reset(seed=42)
        term=False; n=0
        while not term and n<500:
            act=ag.policy(observation=obs)
            cds=[c for c in obs['container_list'] if c['index']==int(act['container_idx'])]
            pred=None
            if cds:
                cd=cds[0]
                try:
                    spec=obs['pool_list'][int(act['item_idx'])]
                    t0=time.perf_counter()
                    tw=Twin(cd,geometry.container_geometry(cd)); tw.sync(cd['packed_items'])
                    q=p.getQuaternionFromEuler(ORNS[int(act['orientation'])])
                    pos=tuple(float(v) for v in act['place_pos'])
                    pred=tw.settles(spec,pos,q,steps_n)
                    tsum+=time.perf_counter()-t0
                    tw.cl.disconnect()
                except Exception as e:
                    pred=None
            obs,r,term,tr,info=env.step(act); n+=1
            st=info.get('status') or {}
            ok = bool(st.get('is_valid')) and bool(st.get('is_placed_safe'))
            if pred is not None and st.get('is_valid'):
                rows.append((pred[0],pred[1],ok))
        env.close()
    return rows,tsum,n

if __name__=='__main__':
    steps_n=int(sys.argv[1]); names=sys.argv[2:]
    scenes=json.load(open('scenes_pool.json'))
    allr=[]; tt=0.0; nn=0
    for nm in names:
        r,ts,n=one(nm,scenes[nm],steps_n); allr+=r; tt+=ts; nn+=n
        print(f'{nm}: {len(r)} verified steps',flush=True)
    print(f'\nmean twin cost = {1000*tt/max(nn,1):.0f} ms/step at {steps_n} settle steps')
    good=[x for x in allr if x[2]]; bad=[x for x in allr if not x[2]]
    print(f'{len(good)} accepted, {len(bad)} rejected by the real validator')
    for lab,arr in (('accepted',good),('rejected',bad)):
        if arr:
            ds=sorted(x[0] for x in arr); as_=sorted(x[1] for x in arr)
            print(f'  {lab:9s} disp med={ds[len(ds)//2]:.4f} p90={ds[int(len(ds)*0.9)]:.4f} max={ds[-1]:.4f}'
                  f' | angle med={as_[len(as_)//2]:.1f} max={as_[-1]:.1f}')
    for td,ta in ((0.03,5),(0.05,8),(0.08,12),(0.12,20)):
        tp=sum(1 for x in bad if x[0]>td or x[1]>ta)
        fp=sum(1 for x in good if x[0]>td or x[1]>ta)
        print(f'  thresh disp>{td} or ang>{ta}: catches {tp}/{len(bad)} topples, '
              f'falsely rejects {fp}/{len(good)} good placements')
