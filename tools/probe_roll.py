import os,sys,json,time,io,contextlib
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from src.ground_handling.env import GroundHandlingEnv
import packer, geometry
scenes=json.load(open('scenes_offline.json'))
for nm in ['O00','O15']:
    cfg=scenes[nm]
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False); env.reset_settings()
        conts=env.get_init_states()['container_list']
        items=env.get_info_for_optimization()
    specs=[{'index':d['index'],'length':d['length'],'width':d['width'],'height':d['height'],
            'mass':d.get('mass',1.0),'is_prioritized':d.get('is_prioritized',False),
            'is_soft':d.get('is_soft',False)} for d in items]
    specs.sort(key=lambda s:-(s['length']*s['width']*s['height']))
    states={c['index']:packer.ContainerState(c) for c in conts}
    t0=time.perf_counter(); cum=[]; placed=0; first_fail=None
    for i,sp in enumerate(specs):
        ts=time.perf_counter()
        best=None;bs=-1e18
        for ci,cs in states.items():
            r=packer.best_placement(cs,sp,time.perf_counter()+30)
            if r and r[2]>bs: bs=r[2]; best=(ci,r[0],r[1])
        if best is None:
            if first_fail is None: first_fail=i
        else:
            ci,pos,orn=best
            states[ci].commit(pos,geometry.half_extents(sp['length'],sp['width'],sp['height'],orn),sp)
            placed+=1
        cum.append(time.perf_counter()-ts)
    tot=time.perf_counter()-t0
    print(f'{nm}: {len(specs)} items -> 1 full rollout = {tot:.1f}s, placed {placed}, '
          f'first failure at item {first_fail}')
    print(f'   cost before first failure: {sum(cum[:first_fail or len(cum)]):.1f}s, '
          f'after: {sum(cum[first_fail or len(cum):]):.1f}s')
    env.close()
