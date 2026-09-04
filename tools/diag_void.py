"""Where does the empty volume actually sit at the end of an episode?"""
import os, sys, json, importlib, io, contextlib
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.environ.get('AGENT_DIR',ROOT+'/agents/submit')); sys.path.insert(0,'.')
import numpy as np
from src.ground_handling.env import GroundHandlingEnv

def run(name,cfg):
    import agent as am; importlib.reload(am)
    am.POLICY_TIME_BUDGET=5.0; am.OPTIMIZE_TIME_BUDGET=60.0
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        env=GroundHandlingEnv(config=cfg,verbose=False,render_mode=None)
        ag=am.Agent(module_path=ROOT+'/agents/submit')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream(); obs,info=env.reset(seed=42)
        term=False; steps=0
        while not term and steps<500:
            obs,r,term,tr,info=env.step(ag.policy(observation=obs)); steps+=1
        c=env.container_manager.containers[0]
        boxes=[]
        for it in c.packed_items:
            pos,orn=it.get_pose(env.client)
            if pos is None: continue
            R=np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3,3)
            hl=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
            boxes.append((np.array(pos),hl))
        dims=(c.length,c.width,c.height,c.thickness,c.cut_x,c.cut_y)
        st=info.get('status')
        env.close()
    return boxes,dims,st,steps

def report(name,cfg):
    boxes,(L,W,H,t,cut_x,cut_y),st,steps=run(name,cfg)
    cell=0.05
    x0,x1=-L/2+t, L/2-t; y0,y1=-W/2+t, W/2-t; z0,z1=t,H-t
    nx=int((x1-x0)/cell); ny=int((y1-y0)/cell); nz=int((z1-z0)/cell)
    occ=np.zeros((nx,ny,nz),bool)
    for pos,h in boxes:
        i0=max(int((pos[0]-h[0]-x0)/cell),0); i1=min(int(np.ceil((pos[0]+h[0]-x0)/cell)),nx)
        j0=max(int((pos[1]-h[1]-y0)/cell),0); j1=min(int(np.ceil((pos[1]+h[1]-y0)/cell)),ny)
        k0=max(int((pos[2]-h[2]-z0)/cell),0); k1=min(int(np.ceil((pos[2]+h[2]-z0)/cell)),nz)
        occ[i0:i1,j0:j1,k0:k1]=True
    sky=np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            k=np.nonzero(occ[i,j])[0]
            sky[i,j]=(k[-1]+1)*cell+z0 if len(k) else z0
    below=0; filled=0
    for i in range(nx):
        for j in range(ny):
            kk=int((sky[i,j]-z0)/cell)
            below+=kk; filled+=occ[i,j,:kk].sum()
    print(f'{name}: n={len(boxes)} steps={steps} status={st}')
    print(f'   skyline mean={sky.mean():.2f} sd={sky.std():.2f} min={sky.min():.2f} max={sky.max():.2f}  ceil={z1:.2f}')
    print(f'   under-skyline occupancy={100*filled/max(below,1):.1f}%  (void under skyline={cell**3*(below-filled):.2f} m3)')
    print(f'   headroom above skyline={cell**3*((nz*nx*ny)-below):.2f} m3   total free={cell**3*(nx*ny*nz-occ.sum()):.2f} m3')
    # per-Y-slice profile (door at y1? report both)
    prof=[f'{sky[:,j].mean():.2f}' for j in range(ny)]
    print('   skyline mean by y (y_lo->y_hi):', ' '.join(prof))
    profx=[f'{sky[i,:].mean():.2f}' for i in range(nx)]
    print('   skyline mean by x (x_lo->x_hi):', ' '.join(profx))

if __name__=='__main__':
    scenes=json.load(open('scenes_pool.json'))
    for k in sys.argv[1:]:
        report(k,scenes[k])
