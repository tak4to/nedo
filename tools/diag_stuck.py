"""Run a scene to the point the agent gets stuck, then ask the REAL
simulator (not the agent's oracle) whether any legal placement remains."""
import os, sys, json, io, contextlib, math, time
import numpy as np
ROOT='/home/takato/comp/nedo'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,'agents','submit'))
from src.ground_handling.env import GroundHandlingEnv
import geometry, packer, agent as agent_module

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def run(name, cfg, pol=5.0, opt=60.0):
    agent_module.POLICY_TIME_BUDGET = pol
    agent_module.OPTIMIZE_TIME_BUDGET = opt
    with quiet():
        env = GroundHandlingEnv(config=cfg, verbose=False)
        ag = agent_module.Agent(module_path='')
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        term=False; steps=0; last={}
        while not term and steps<500:
            a = ag.policy(observation=obs)
            obs, r, term, trunc, last = env.step(a)
            steps+=1
    return env, obs, last, steps

def report_layout(env):
    c = env.container_manager.containers[0]
    print(f'container L={c.length} W={c.width} H={c.height} vol={c.volume:.3f}')
    rows=[]
    for it in c.packed_items:
        pos,orn = it.get_pose(env.client)
        R=np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3,3)
        h=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
        rows.append((pos[0]-c.center[0],pos[1],pos[2],h[0],h[1],h[2],it.volume))
    rows.sort(key=lambda r:(r[2],r[1]))
    print(f'{len(rows)} boxes, total vol {sum(r[6] for r in rows):.3f} '
          f'({100*sum(r[6] for r in rows)/c.volume:.1f}% of container)')
    for r in rows:
        print(f'  x[{r[0]-r[3]:+.3f},{r[0]+r[3]:+.3f}] y[{r[1]-r[4]:+.3f},{r[1]+r[4]:+.3f}] '
              f'z[{r[2]-r[5]:.3f},{r[2]+r[5]:.3f}]')
    return rows

def free_volume_map(env, step=0.05):
    """Occupancy grid of the usable interior."""
    c = env.container_manager.containers[0]
    boxes=[]
    for it in c.packed_items:
        pos,orn = it.get_pose(env.client)
        R=np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3,3)
        h=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
        boxes.append((np.array(pos)-np.array([c.center[0],0,0]),h))
    n_vecs=np.array(c.n_vecs); pts=np.array(c.points)
    xs=np.arange(-c.length/2, c.length/2, step)
    ys=np.arange(-c.width/2, c.width/2, step)
    zs=np.arange(0, c.height, step)
    free=0; inside=0
    zprofile={}
    for z in zs:
        for y in ys:
            for x in xs:
                g=np.array([x+c.center[0],y,z])
                if np.any(np.sum(n_vecs*(g-pts),axis=1) > 0):
                    continue
                inside+=1
                occ=any(abs(x-b[0][0])<b[1][0] and abs(y-b[0][1])<b[1][1] and abs(z-b[0][2])<b[1][2]
                        for b in boxes)
                if not occ:
                    free+=1
                    zprofile[round(z,2)]=zprofile.get(round(z,2),0)+1
    print(f'grid {step}m: inside cells {inside}, free {free} ({100*free/inside:.1f}%)')
    print('free cells by z:', {k:v for k,v in sorted(zprofile.items())})
    return free/inside

def exhaustive_real_check(env, obs, xstep=0.02, ystep=0.02, max_tries=200000):
    """Brute-force the pool items against the REAL validator."""
    from src.ground_handling.utils import get_half_ext
    c = env.container_manager.containers[0]
    pool = env.stream_manager.visible_pool
    found=[]
    tried=0
    for pi,item in enumerate(pool):
        for orn in range(6):
            hx,hy,hz = get_half_ext([item.length,item.width,item.height],orn)
            xs=np.arange(-c.length/2+c.thickness+hx, c.length/2-c.thickness-hx+1e-9, xstep)
            ys=np.arange(-c.width/2+hy, c.width/2-c.thickness-hy+1e-9, ystep)
            # candidate z: rest on floor or on any box top
            tops={c.thickness}
            for it in c.packed_items:
                pos,o = it.get_pose(env.client)
                R=np.array(env.client.getMatrixFromQuaternion(o)).reshape(3,3)
                h=np.abs(R)@np.array([it.length/2,it.width/2,it.height/2])
                tops.add(round(pos[2]+h[2],4))
            for z0 in sorted(tops):
                z = z0+hz
                for x in xs:
                    for y in ys:
                        tried+=1
                        if tried>max_tries: 
                            print('  (hit try cap)'); return found
                        gp=(x+c.center[0], y, z)
                        with quiet():
                            if not env.validator.check_inclusion(c,item,np.array(gp),orn):
                                continue
                            if not env.validator.check_transport_path(c,item,gp,orn):
                                continue
                        found.append((pi,item.index,orn,gp,z0))
                        print(f'  REAL-VALID: pool{pi} item{item.index} '
                              f'{item.length}x{item.width}x{item.height} orn{orn} '
                              f'at local ({x:.3f},{y:.3f},{z:.3f}) rest_on_z={z0:.3f}')
                        if len(found)>=15: return found
    print(f'  tried {tried} placements')
    return found

if __name__=='__main__':
    scenes=json.load(open('scenes.json'))
    name=sys.argv[1] if len(sys.argv)>1 else 'C_lookahead1'
    cfg=scenes[name]
    cfg['containers']['container_list']=cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers']=1
    env,obs,last,steps=run(name,cfg)
    print(f'=== {name} stopped after {steps} steps, status={last.get("status")}')
    report_layout(env)
    free_volume_map(env)
    print('--- exhaustive check with the REAL validator ---')
    f=exhaustive_real_check(env,obs)
    print(f'REAL valid placements the agent missed: {len(f)}')
