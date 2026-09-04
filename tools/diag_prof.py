import os,sys,json,time,cProfile,pstats,io
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit')); sys.path.insert(0,'.')
import packer
from diag_stuck import run
nm=sys.argv[1]
cfg=json.load(open('scenes_pool.json'))[nm]
cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
cfg['camera']['num_containers']=1
env,obs,last,steps=run(nm,cfg)
cds=env.container_manager.get_item_info_in_containers()
cs=packer.ContainerState(cds[0])
it=env.stream_manager.visible_pool[0]
spec=dict(index=it.index,length=it.length,width=it.width,height=it.height,mass=it.mass,
          is_prioritized=it.is_prioritized,is_soft=it.is_soft)
print('boxes',len(cs.boxes),'static',len(cs.static_obstacles))
t=time.perf_counter(); c=packer._collect(cs,spec,False,packer._xy_candidates); t1=time.perf_counter()-t
print(f'_collect(extreme): {t1*1000:.0f}ms  {len(c)} cands')
t=time.perf_counter(); r=packer._first_valid(cs,c,None); t2=time.perf_counter()-t
print(f'_first_valid: {t2*1000:.0f}ms -> {"ok" if r else "none"}')
t=time.perf_counter(); r=packer.best_placement(cs,spec,time.perf_counter()+30); t3=time.perf_counter()-t
print(f'best_placement total: {t3*1000:.0f}ms -> {"ok" if r else "none"}')
pr=cProfile.Profile(); pr.enable()
packer.best_placement(cs,spec,time.perf_counter()+30)
pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('tottime').print_stats(12); print(s.getvalue())
env.close()
