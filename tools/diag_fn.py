"""At the stuck state: of the placements the agent REJECTS, how many does
the real validator accept, and which of the agent's two gates rejected them?"""
import os,sys,json,random,io,contextlib
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from src.ground_handling.utils import get_half_ext
import geometry, packer
from diag_stuck import run, quiet

def scan(env, sample=400, xstep=0.04, ystep=0.04, seed=0):
    c=env.container_manager.containers[0]
    cd=env.container_manager.get_item_info_in_containers()[0]
    cs=packer.ContainerState(cd); g=cs.geom; obst=cs.obstacle_boxes()
    pool=env.stream_manager.visible_pool
    tops={c.thickness}
    for it in c.packed_items:
        pos,o=it.get_pose(env.client)
        R=np.array(env.client.getMatrixFromQuaternion(o)).reshape(3,3)
        h=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
        tops.add(round(pos[2]+h[2],4))
    buckets={'geom':[], 'support':[], 'accept':[]}
    for item in pool:
        for orn in range(6):
            hx,hy,hz=get_half_ext([item.length,item.width,item.height],orn)
            for x in np.arange(-c.length/2+c.thickness+hx, c.length/2-c.thickness-hx+1e-9, xstep):
                for y in np.arange(-c.width/2+hy, c.width/2-c.thickness-hy+1e-9, ystep):
                    spec={'length':item.length,'width':item.width,'height':item.height,
                          'mass':item.mass,'is_prioritized':item.is_prioritized,'is_soft':item.is_soft}
                    st_top, sup = packer._landing(cs, float(x), float(y), hx, hy)
                    bottom = packer._bottom_for_support(g, st_top, sup, hz)
                    z = bottom + hz
                    if bottom+2*hz > g['z_hi']: continue
                    loc=(float(x),float(y),float(z))
                    geom_ok = (geometry.check_inclusion(g,loc,(hx,hy,hz)) and
                               geometry.check_transport_path(g,obst,loc,(hx,hy,hz)))
                    sup_ok = (sup is None) or packer._footprint_supported(
                        float(x),float(y),2*hx,2*hy,sup)
                    rec=(item,orn,loc)
                    if not geom_ok: buckets['geom'].append(rec)
                    elif not sup_ok: buckets['support'].append(rec)
                    else: buckets['accept'].append(rec)
    rng=random.Random(seed)
    out={}
    for k in ('geom','support'):
        pop=buckets[k]
        s=rng.sample(pop, min(sample, len(pop)))
        fn=0
        for (item,orn,loc) in s:
            gp=(loc[0]+c.center[0],loc[1],loc[2])
            with quiet():
                if env.validator.check_inclusion(c,item,np.array(gp),orn) and \
                   env.validator.check_transport_path(c,item,gp,orn):
                    fn+=1
        out[k]=(len(pop), len(s), fn)
    out['accept']=(len(buckets['accept']),0,0)
    return out

if __name__=='__main__':
    nm=sys.argv[1]; f=sys.argv[2] if len(sys.argv)>2 else 'scenes_pool.json'
    cfg=json.load(open(f))[nm]
    cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers']=1
    env,obs,last,steps=run(nm,cfg)
    c=env.container_manager.containers[0]
    vol=sum(i.volume for i in c.packed_items)
    print(f'{nm}: {steps}手で終了, {len(c.packed_items)}箱, 体積{100*vol/c.volume:.1f}%')
    r=scan(env)
    print(f"  エージェントが受理した候補: {r['accept'][0]}")
    for k,lab in (('geom','幾何オラクルが棄却'),('support','支持判定が棄却')):
        n,s,fn=r[k]
        print(f"  {lab}: {n} 件中 {s} 件を実バリデータで検査 -> "
              f"実は合法だったもの {fn} 件 ({100*fn/max(s,1):.1f}%)")
