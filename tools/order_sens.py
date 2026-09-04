"""How much does the offline item ORDER actually matter?
optimize() can only return a permutation, so this bounds what any amount
of offline search over orders can possibly buy."""
import os,sys,json,random,time
from concurrent.futures import ProcessPoolExecutor
ROOT='/home/takato/comp/nedo'

def _one(args):
    name,cfg,mode,seed = args
    sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    import patches; patches.apply3('com60')
    import packer; packer.W_FLAT=600.0
    import harness, agent as am, io, contextlib
    from src.ground_handling.env import GroundHandlingEnv
    am.POLICY_TIME_BUDGET=5.0; am.OPTIMIZE_TIME_BUDGET=60.0
    cfg=json.loads(json.dumps(cfg)); cfg['agent']['optimize']=True
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False)
        ag=am.Agent(module_path='')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        items=env.get_info_for_optimization()
        idx=[d['index'] for d in items]
        if mode=='opt':   order=ag.optimize(items)
        elif mode=='id':  order=list(idx)
        else:
            r=random.Random(seed); order=list(idx); r.shuffle(order)
        env.set_item_order(order); env.reset_item_stream()
        obs,info=env.reset(seed=42)
        term=False; n=0
        while not term and n<500:
            obs,rr,term,tr,last=env.step(ag.policy(observation=obs)); n+=1
        rep=env.evaluate()
        packed=sum(len(c.packed_items) for c in env.container_manager.containers)
        tot=env.num_total_items
        env.close()
    ceil_=harness.ceiling_fill(cfg)
    return (name,mode,seed,100*rep['fill_score']/ceil_,100*packed/tot)

if __name__=='__main__':
    scenes=json.load(open('scenes_pool.json'))
    names=['P00','P05','P13','P19','P26']
    jobs=[]
    for n in names:
        jobs.append((n,scenes[n],'opt',0)); jobs.append((n,scenes[n],'id',0))
        for s in range(6): jobs.append((n,scenes[n],'rand',s))
    with ProcessPoolExecutor(max_workers=14) as ex:
        rows=list(ex.map(_one,jobs))
    import statistics as st
    by={}
    for (n,m,s,norm,pct) in rows: by.setdefault(n,[]).append((m,s,norm,pct))
    print(f"{'scene':6s} {'optimize':>9s} {'identity':>9s} {'rand mean':>10s} {'rand min':>9s} {'rand max':>9s} {'rand sd':>8s}")
    for n in names:
        d=dict(((m,s),(norm,pct)) for (m,s,norm,pct) in by[n])
        rs=[v[0] for (m,s),v in d.items() if m=='rand']
        print(f"{n:6s} {d[('opt',0)][0]:9.1f} {d[('id',0)][0]:9.1f} {st.mean(rs):10.1f} "
              f"{min(rs):9.1f} {max(rs):9.1f} {st.pstdev(rs):8.2f}")
