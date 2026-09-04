"""A/B a set of constant overrides across scenes, in parallel."""
import os,sys,json,argparse,io,contextlib,time
from concurrent.futures import ProcessPoolExecutor
ROOT='/home/takato/comp/nedo'

def _one(args):
    name, cfg, overrides, pol, opt, patch, agent_dir = args
    os.environ['AGENT_DIR'] = agent_dir or os.path.join(ROOT,'agents','submit')
    sys.path.insert(0,ROOT); sys.path.insert(0, os.environ['AGENT_DIR'])
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    import geometry, packer
    for mod, k, v in overrides:
        m = {'g':geometry,'p':packer}[mod]
        setattr(m, k, v)
        if mod=='g' and hasattr(packer,k):
            setattr(packer,k,v)
    if patch and patch != 'none':
        import patches; patches.apply5(patch)
    import harness
    return harness.run_scene(name, cfg, policy_budget=pol, optimize_budget=opt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenes',default='scenes.json')
    ap.add_argument('--only',default=None)
    ap.add_argument('--set',action='append',default=[])
    ap.add_argument('--policy-budget',type=float,default=5.0)
    ap.add_argument('--optimize-budget',type=float,default=60.0)
    ap.add_argument('--tag',default='base')
    ap.add_argument('--jobs',type=int,default=7)
    ap.add_argument('--patch',default='none')
    ap.add_argument('--agent-dir',default=None)
    a=ap.parse_args()
    ov=[]
    for s in a.set:
        mod,rest=s.split(':',1); k,v=rest.split('=',1)
        ov.append((mod,k,float(v)))
    scenes=json.load(open(a.scenes))
    if a.only:
        ks=a.only.split(','); scenes={k:v for k,v in scenes.items() if k in ks}
    jobs=[(n,c,ov,a.policy_budget,a.optimize_budget,a.patch,a.agent_dir) for n,c in scenes.items()]
    t0=time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows=list(ex.map(_one,jobs))
    rows.sort(key=lambda r:r['scene'])
    print(f'### {a.tag}  overrides={a.set} patch={a.patch}')
    def comp(r):
        return (r['fill'] + r.get('cog_score',0) + r.get('stability_score',0)
                + r.get('placement_score',0) + r.get('soft_item_score',0)) / 5.0
    for r in rows:
        print(f"  {r['scene']:16s} fill={r['fill']:5.1f} cog={r.get('cog_score',0):5.1f} "
              f"stab={r.get('stability_score',0):5.1f} place={r.get('placement_score',0):5.1f} "
              f"soft={r.get('soft_item_score',0):5.1f} | comp={comp(r):5.1f} "
              f"packed={r['pct']:4.1f}% disp={r.get('shake_mean_disp',0):.3f}")
    n=len(rows)
    print(f"  == norm%={sum(r['norm'] for r in rows)/n:5.2f} fill={sum(r['fill'] for r in rows)/n:5.2f} "
          f"cog={sum(r.get('cog_score',0) for r in rows)/n:5.2f} "
          f"stab={sum(r.get('stability_score',0) for r in rows)/n:5.2f} "
          f"place={sum(r.get('placement_score',0) for r in rows)/n:5.2f} "
          f"soft={sum(r.get('soft_item_score',0) for r in rows)/n:5.2f} "
          f"| COMPOSITE={sum(comp(r) for r in rows)/n:5.2f} "
          f"packed={sum(r['pct'] for r in rows)/n:4.1f}%  [{time.time()-t0:.0f}s]")
    json.dump(rows,open(f'ab_{a.tag}.json','w'),indent=1)

if __name__=='__main__':
    main()
