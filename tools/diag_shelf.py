"""Is the under-shelf volume actually reachable, and does _landing hide it?"""
import os,sys,json
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit')); sys.path.insert(0,'.')
import packer, geometry
from diag_stuck import run
f,nm=sys.argv[1],sys.argv[2]
cfg=json.load(open(f))[nm]
env,obs,last,steps=run(nm,cfg)
cd=env.container_manager.get_item_info_in_containers()[0]
cs=packer.ContainerState(cd); g=cs.geom; obst=cs.obstacle_boxes()
print('static boxes:', [(tuple(round(v,3) for v in c), tuple(round(v,3) for v in h)) for c,h in g['static_boxes']])
print('shelf_struct_z=%.3f floor_struct_z=%.3f GAP=%.3f SUPPORT_LIFT=%.3f'%(
    g['shelf_struct_z'], g['floor_struct_z'], geometry.GAP, packer.SUPPORT_LIFT))
it=env.stream_manager.visible_pool[0]
spec=dict(index=it.index,length=it.length,width=it.width,height=it.height,mass=it.mass,
          is_prioritized=it.is_prioritized,is_soft=it.is_soft)
print('probe item lwh=(%.2f,%.2f,%.2f)'%(it.length,it.width,it.height))
hx,hy,hz=geometry.half_extents(it.length,it.width,it.height,0)
ok_under=ok_on=0; tried=0
for cx in [x/100 for x in range(-80,81,10)]:
    for cy in [y/100 for y in range(5,60,5)]:
        for bottom,label in ((g['floor_struct_z']+geometry.FLOOR_LIFT,'under'),):
            pos=(cx,cy,bottom+hz); tried+=1
            if bottom+2*hz > g['shelf_struct_z']-geometry.thickness_dummy if False else False: pass
            if geometry.validate_placement(g,obst,pos,(hx,hy,hz)): ok_under+=1
for cx in [x/100 for x in range(-80,81,10)]:
    for cy in [y/100 for y in range(5,60,5)]:
        for lift in (packer.SUPPORT_LIFT, 0.035, 0.05):
            bottom=g['shelf_struct_z']+lift
            pos=(cx,cy,bottom+hz)
            if geometry.validate_placement(g,obst,pos,(hx,hy,hz)):
                ok_on+=1
                if lift==packer.SUPPORT_LIFT: print('   ON-SHELF ok at current lift', cx,cy)
print(f'under-shelf (floor level, y>0): {ok_under}/{tried} positions pass the full oracle')
for lift in (packer.SUPPORT_LIFT,0.030,0.035,0.05,0.08):
    n=0; t=0
    for cx in [x/100 for x in range(-80,81,10)]:
        for cy in [y/100 for y in range(5,60,5)]:
            bottom=g['shelf_struct_z']+lift
            t+=1
            if geometry.validate_placement(g,obst,(cx,cy,bottom+hz),(hx,hy,hz)): n+=1
    print(f'  on-shelf with SUPPORT_LIFT={lift:.3f}: {n}/{t} valid')
# what does _landing say for an under-shelf candidate?
st,sup=packer._landing(cs,0.0,0.30,hx,hy)
print(f'_landing(0.0,0.30) -> support_top={st:.3f} supporters={0 if sup is None else len(sup)} static={None if sup is None else any(b["is_static"] for b in sup)}')
env.close()
