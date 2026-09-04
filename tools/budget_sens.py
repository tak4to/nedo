import os,sys,json,time
from concurrent.futures import ProcessPoolExecutor
ROOT='/home/takato/comp/nedo'
def _one(args):
    name,cfg,ob = args
    sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    import patches; patches.apply3('com60')
    import packer; packer.W_FLAT=600.0
    import harness, io, contextlib
    cfg=json.loads(json.dumps(cfg)); cfg['agent']['optimize']=True
    r=harness.run_scene(name,cfg,policy_budget=5.0,optimize_budget=ob)
    return (name,ob,r['norm'],r['pct'])
if __name__=='__main__':
    scenes=json.load(open('scenes_pool.json'))
    names=['P00','P05','P13','P19','P26']
    jobs=[(n,scenes[n],ob) for n in names for ob in (15,60,150)]
    with ProcessPoolExecutor(max_workers=14) as ex:
        rows=list(ex.map(_one,jobs))
    d={(n,ob):(nm,pc) for (n,ob,nm,pc) in rows}
    print(f"{'scene':6s} {'opt=15s':>9s} {'opt=60s':>9s} {'opt=150s':>9s}")
    for n in names:
        print(f"{n:6s} {d[(n,15)][0]:9.1f} {d[(n,60)][0]:9.1f} {d[(n,150)][0]:9.1f}")
    import statistics as st
    for ob in (15,60,150):
        print(f"  mean opt={ob:3d}s : norm {st.mean(d[(n,ob)][0] for n in names):5.1f}  "
              f"packed {st.mean(d[(n,ob)][1] for n in names):5.1f}%")
