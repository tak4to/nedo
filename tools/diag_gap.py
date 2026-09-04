"""At the stuck state, does packer's own search find what brute force finds?"""
import os,sys,json,time
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit')); sys.path.insert(0,'.')
import geometry, packer
from diag_stuck import run

def probe(nm, f='scenes_pool.json'):
    cfg=json.load(open(f))[nm]
    env,obs,last,steps=run(nm,cfg)
    cds=env.container_manager.get_item_info_in_containers()
    states={c['index']:packer.ContainerState(c) for c in cds}
    pool=[(i,dict(index=it.index,length=it.length,width=it.width,height=it.height,
                  mass=it.mass,is_prioritized=it.is_prioritized,is_soft=it.is_soft))
          for i,it in enumerate(env.stream_manager.visible_pool)]
    print(f'{nm}: steps={steps} boxes={len(cds[0]["packed_items"])} pool={len(pool)}')
    for relax in (False,True):
        for req in (True,False):
            t=time.perf_counter()
            a=packer.choose_action(states,pool,time.perf_counter()+20.0,relax=relax,require_support=req)
            print(f'   choose_action relax={relax} require_support={req}: '
                  f'{"FOUND" if a else "none"}  ({time.perf_counter()-t:.1f}s)')
    # now: y-grid fallback
    cs=states[cds[0]['index']]
    for pi,spec in pool:
        for step in (0.05,0.03):
            def xyg(c,fx,fy,walls,_s=step):
                g=c.geom
                xs=[g['x_lo']+i*_s for i in range(max(int((g['x_hi']-fx-g['x_lo'])/_s),0)+1)]+[g['x_hi']-fx]
                ys=[g['y_lo']+fy+i*_s for i in range(max(int((g['y_hi']-g['y_lo']-fy)/_s),0)+1)]+[g['y_hi']]
                return [(x,y,None) for y in sorted(ys,reverse=True) for x in xs]
            cands=packer._collect(cs,spec,True,xyg)
            r=packer._first_valid(cs,cands,time.perf_counter()+30.0)
            print(f'   XY-grid step={step} item={spec["index"]}: {"FOUND "+str(r[0]) if r else "none"} ({len(cands)} cands)')
    env.close()

if __name__=='__main__':
    args=sys.argv[1:]
    f='scenes_pool.json'
    if args and args[0].endswith('.json'):
        f=args.pop(0)
    for nm in args:
        probe(nm, f)
