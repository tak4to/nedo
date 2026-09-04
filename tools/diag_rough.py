"""Skyline roughness at the stuck state, for a given W_LEFT."""
import os,sys,json
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
import packer
packer.W_LEFT=float(sys.argv[3]) if len(sys.argv)>3 else packer.W_LEFT
from diag_stuck import run
nm=sys.argv[1]; f=sys.argv[2]
cfg=json.load(open(f))[nm]
cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
cfg['camera']['num_containers']=1
env,obs,last,steps=run(nm,cfg)
c=env.container_manager.containers[0]
boxes=[]
for it in c.packed_items:
    pos,o=it.get_pose(env.client)
    R=np.array(env.client.getMatrixFromQuaternion(o)).reshape(3,3)
    h=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
    boxes.append((pos[0]-c.center[0],pos[1],pos[2],h[0],h[1],h[2]))
xs=np.arange(-c.length/2+c.thickness, c.length/2-c.thickness, 0.05)
ys=np.arange(-c.width/2, c.width/2-c.thickness, 0.05)
H=np.full((len(ys),len(xs)), c.thickness)
for j,y in enumerate(ys):
    for i,x in enumerate(xs):
        for (bx,by,bz,hx,hy,hz) in boxes:
            if abs(x-bx)<hx and abs(y-by)<hy: H[j,i]=max(H[j,i], bz+hz)
vol=sum(i.volume for i in c.packed_items)
print(f"W_LEFT={packer.W_LEFT:6.1f}  {nm}: {steps}手 {len(boxes)}箱 体積{100*vol/c.volume:4.1f}%  "
      f"スカイライン 平均{H.mean():.2f} sd{H.std():.2f} 最大{H.max():.2f}")
