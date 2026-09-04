import os,sys,json
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit')); sys.path.insert(0,'.')
import packer, geometry
from diag_stuck import run
nm=sys.argv[2]; f=sys.argv[1]
cfg=json.load(open(f))[nm]
print('containers:',[{k:v for k,v in c.items() if k in ('index','length','width','height','thickness','cut_x','cut_y','require_shelf','is_prioritized')} for c in cfg['containers']['container_list']])
print('n_items',len(cfg['item_stream']['item_list']),'look_ahead',cfg['item_stream']['look_ahead'],'optimize',cfg['agent']['optimize'])
env,obs,last,steps=run(nm,cfg)
for cd in env.container_manager.get_item_info_in_containers():
    cs=packer.ContainerState(cd); g=cs.geom
    print(f"--- container {cd['index']}  prio={cd['is_prioritized']} shelf={cd['shelf']} boxes={len(cs.boxes)}")
    print(f"    design box x[{g['x_lo']:.2f},{g['x_hi']:.2f}] y[{g['y_lo']:.2f},{g['y_hi']:.2f}] z[{g['z_lo']:.2f},{g['z_hi']:.2f}]")
    for b in sorted(cs.boxes,key=lambda b:-b['center'][2]):
        c=b['center']; h=b['half']
        print(f"      c=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) half=({h[0]:.2f},{h[1]:.2f},{h[2]:.2f}) top={c[2]+h[2]:.2f} soft={b['is_soft']} prio={b['is_prioritized']}")
print('pool:')
for it in env.stream_manager.visible_pool:
    print(f'   idx={it.index} lwh=({it.length:.2f},{it.width:.2f},{it.height:.2f}) m={it.mass} soft={it.is_soft} prio={it.is_prioritized}')
env.close()
