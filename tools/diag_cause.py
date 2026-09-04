"""Why is every candidate rejected at the stuck state? Split the geometric
rejections into inclusion / Y-sweep (push in from the door) / X-sweep."""
import os,sys,json
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from src.ground_handling.utils import get_half_ext
import geometry, packer
from diag_stuck import run

def why(env, xstep=0.03, ystep=0.03):
    c=env.container_manager.containers[0]
    cd=env.container_manager.get_item_info_in_containers()[0]
    cs=packer.ContainerState(cd); g=cs.geom; obst=cs.obstacle_boxes()
    cnt={'inclusion':0,'y_sweep':0,'x_sweep':0,'no_support':0,'accept':0,'too_tall':0}
    for item in env.stream_manager.visible_pool:
        for orn in range(6):
            hx,hy,hz=get_half_ext([item.length,item.width,item.height],orn)
            he=(hx,hy,hz)
            for x in np.arange(-c.length/2+c.thickness+hx, c.length/2-c.thickness-hx+1e-9, xstep):
                for y in np.arange(-c.width/2+hy, c.width/2-c.thickness-hy+1e-9, ystep):
                    st_top,sup=packer._landing(cs,float(x),float(y),hx,hy)
                    bottom=packer._bottom_for_support(g,st_top,sup,hz)
                    if bottom+2*hz>g['z_hi']: cnt['too_tall']+=1; continue
                    loc=(float(x),float(y),bottom+hz)
                    if not geometry.check_inclusion(g,loc,he): cnt['inclusion']+=1; continue
                    x_min,x_max=geometry.x_transport_range(g,hx)
                    rel_x=min(max(loc[0],x_min),x_max)
                    eff=geometry.resting_eff_start_z(g,loc[2],hz)
                    z_cap=g['height']+g['buffer']-g['thickness']-hz-geometry.START_MARGIN
                    rel_z=min(z_cap, loc[2]+eff)
                    sy=-g['width']/2.0
                    if not geometry._swept_clear(obst,1,(rel_x,sy,rel_z),he,sy,loc[1],geometry.GAP):
                        cnt['y_sweep']+=1; continue
                    if not geometry._swept_clear(obst,0,(rel_x,loc[1],rel_z),he,rel_x,loc[0],geometry.GAP):
                        cnt['x_sweep']+=1; continue
                    if sup is not None and not packer._footprint_supported(loc[0],loc[1],2*hx,2*hy,sup):
                        cnt['no_support']+=1; continue
                    cnt['accept']+=1
    return cnt

if __name__=='__main__':
    nm=sys.argv[1]; f=sys.argv[2] if len(sys.argv)>2 else 'scenes_pool.json'
    cfg=json.load(open(f))[nm]
    cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers']=1
    env,obs,last,steps=run(nm,cfg)
    c=env.container_manager.containers[0]
    vol=sum(i.volume for i in c.packed_items)
    cnt=why(env); tot=sum(cnt.values())
    print(f'{nm}: {steps}手, {len(c.packed_items)}箱, 体積{100*vol/c.volume:.1f}%  '
          f'(候補 {tot} 件を分類)')
    lab={'inclusion':'内包判定NG','y_sweep':'扉からの押し込み(Y)で衝突','x_sweep':'横スライド(X)で衝突',
         'no_support':'支持不足','too_tall':'高さ超過','accept':'受理'}
    for k in ('accept','y_sweep','x_sweep','inclusion','too_tall','no_support'):
        print(f'   {lab[k]:26s} {cnt[k]:7d}  ({100*cnt[k]/tot:5.1f}%)')
