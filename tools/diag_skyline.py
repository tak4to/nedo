"""Height map of the stuck state: is the container tall-and-spiky (needs
level discipline) or is the leftover space just too narrow?"""
import os,sys,json
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from diag_stuck import run
if __name__=='__main__':
    nm=sys.argv[1]; f=sys.argv[2] if len(sys.argv)>2 else 'scenes_pool.json'
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
    ceil=c.height-c.thickness
    xs=np.arange(-c.length/2+c.thickness, c.length/2-c.thickness, 0.05)
    ys=np.arange(-c.width/2, c.width/2-c.thickness, 0.05)
    H=np.full((len(ys),len(xs)), c.thickness)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            for (bx,by,bz,hx,hy,hz) in boxes:
                if abs(x-bx)<hx and abs(y-by)<hy:
                    H[j,i]=max(H[j,i], bz+hz)
    print(f'{nm}: {steps}手, {len(boxes)}箱, 天井 z={ceil:.2f}')
    print(f'  スカイライン高さ: 平均 {H.mean():.2f}, 中央値 {np.median(H):.2f}, '
          f'最大 {H.max():.2f}, 標準偏差 {H.std():.2f}')
    print(f'  天井までの残り高さ: 平均 {ceil-H.mean():.2f} m')
    for thr in (0.20,0.24,0.27,0.40):
        frac=100*(ceil-H > thr).mean()
        print(f'    残り高さ > {thr:.2f}m の面積割合: {frac:5.1f}%')
    print('  高さマップ (行=Y 奥→手前, 列=X 左→右, 単位cm):')
    for j in range(len(ys)-1,-1,-1):
        print('   ', ' '.join(f'{int(H[j,i]*100):3d}' for i in range(len(xs))))
