"""Is the fast _landing byte-identical to the shipped one?"""
import os,sys,json,random,time
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit')); sys.path.insert(0,'.')
import packer
from diag_stuck import run
orig = packer._landing
import patches; patches._mk_fastland(); fast = packer._landing
packer._landing = orig

rng=random.Random(0)
scenes=json.load(open('scenes_pool.json'))
bad=0; n=0; t_o=t_f=0.0
for nm in sys.argv[1:]:
    cfg=scenes[nm]
    cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers']=1
    env,obs,last,steps=run(nm,cfg)
    cs=packer.ContainerState(env.container_manager.get_item_info_in_containers()[0])
    g=cs.geom
    qs=[(rng.uniform(g['x_lo'],g['x_hi']), rng.uniform(g['y_lo'],g['y_hi']),
         rng.uniform(0.1,0.4), rng.uniform(0.1,0.35)) for _ in range(30000)]
    t=time.perf_counter()
    ro=[orig(cs,*q) for q in qs]; t_o+=time.perf_counter()-t
    t=time.perf_counter()
    rf=[fast(cs,*q) for q in qs]; t_f+=time.perf_counter()-t
    for a,b in zip(ro,rf):
        n+=1
        sa = None if a[1] is None else sorted(id(x) for x in a[1])
        sb = None if b[1] is None else sorted(id(x) for x in b[1])
        if abs(a[0]-b[0])>1e-12 or sa!=sb: bad+=1
    env.close()
print(f'compared {n} queries, mismatches={bad}')
print(f'orig {t_o*1000:.0f}ms  fast {t_f*1000:.0f}ms  speedup {t_o/t_f:.2f}x')
