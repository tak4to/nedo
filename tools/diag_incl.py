"""Which inclusion plane rejects the items we placed, and by how much?"""
import os,sys,json,importlib,io,contextlib,collections
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.environ.get('AGENT_DIR',ROOT+'/agents/submit')); sys.path.insert(0,'.')
import numpy as np
from src.ground_handling.env import GroundHandlingEnv

def run(name,cfg):
    import agent as am; importlib.reload(am)
    am.POLICY_TIME_BUDGET=5.0; am.OPTIMIZE_TIME_BUDGET=60.0
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False,render_mode=None)
        ag=am.Agent(module_path=ROOT+'/agents/submit')
        env.reset_settings(); ag.get_init_states(env.get_init_states())
        if cfg['agent']['optimize']:
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream(); obs,info=env.reset(seed=42)
        term=False;n=0
        while not term and n<500:
            obs,r,term,tr,info=env.step(ag.policy(observation=obs)); n+=1
        rows=[]
        for c in env.container_manager.containers:
            nv=np.array(c.n_vecs); pts=np.array(c.points)
            for it in c.packed_items:
                pos,orn=it.get_pose(env.client)
                if pos is None: continue
                R=np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3,3)
                hl,hw,hh=it.length/2,it.width/2,it.height/2
                lc=np.array([[sx*hl,sy*hw,sz*hh] for sx in(1,-1) for sy in(1,-1) for sz in(1,-1)])
                gc=lc@R.T+np.array(pos)
                worst=-9e9; wi=-1
                for corner in gc:
                    d=np.sum(nv*(corner-pts),axis=1)
                    j=int(np.argmax(d))
                    if d[j]>worst: worst=float(d[j]); wi=j
                bz=float(min(c[2] for c in gc))
                rows.append((wi,worst,it.volume,bz))
        planes=[(i,tuple(round(float(v),3) for v in nv[i]),round(float(pts[i][2]),3)) for i in range(len(nv))]
        env.close()
    return rows,planes

if __name__=='__main__':
    scenes=json.load(open(sys.argv[1]))
    names=sys.argv[2:]
    agg=collections.Counter(); vol=collections.Counter(); tot=0; totv=0.0; planes=None
    margins=[-0.005,0.0,0.005,0.01,0.02]
    cnt={m:0 for m in margins}; cv={m:0.0 for m in margins}; allrows=[]
    for nm in names:
        rows,planes=run(nm,scenes[nm])
        for wi,worst,v,bz in rows:
            tot+=1; totv+=v
            allrows.append((wi,worst,v,bz))
            if worst>-0.005: agg[wi]+=1; vol[wi]+=v
            for m in margins:
                if worst<=m: cnt[m]+=1; cv[m]+=v
    print('planes (idx, outward normal, point z):')
    for p in planes: print('   ',p)
    print(f'\nplaced={tot} volume={totv:.3f}')
    print('rejected at margin -0.005, by plane:')
    for i,c in agg.most_common():
        print(f'   plane {i} normal={planes[i][1]}: {c} items, {vol[i]:.3f} m3 ({100*vol[i]/totv:.1f}% of placed volume)')
    floor=[r for r in allrows if r[3]<0.045]
    up=[r for r in allrows if r[3]>=0.045]
    fv=sum(r[2] for r in floor); uv=sum(r[2] for r in up)
    fc=sum(1 for r in floor if r[1]<=-0.005); uc=sum(1 for r in up if r[1]<=-0.005)
    print(f'\nsettled bottom < 0.045 (floor-resting): {len(floor)} items, {fv:.3f} m3 -> counted {fc}')
    print(f'settled bottom >= 0.045 (on something) : {len(up)} items, {uv:.3f} m3 -> counted {uc}')
    print('\ncounted volume vs margin:')
    for m in margins:
        print(f'   margin {m:+.3f}: {cnt[m]}/{tot} items, {cv[m]:.3f} m3 ({100*cv[m]/totv:.1f}%)')
