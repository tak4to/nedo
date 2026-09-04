"""2コンテナのとき、荷物は両方に分散しているか？"""
import os,sys,json
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from diag_stuck import run
sc=json.load(open(sys.argv[2] if len(sys.argv)>2 else 'scenes_pool.json'))
for nm in sys.argv[1].split(','):
    cfg=sc[nm]
    if len(cfg['containers']['container_list'])<2: continue
    env,obs,last,steps=run(nm,cfg)
    cs=env.container_manager.containers
    tot=env.num_total_items
    counts=[len(c.packed_items) for c in cs]
    vols=[sum(i.volume for i in c.packed_items) for c in cs]
    caps=[c.volume for c in cs]
    print(f"{nm}: {steps}手, 全{tot}個 -> 積載 {sum(counts)} ({100*sum(counts)/tot:.1f}%)")
    for i,c in enumerate(cs):
        print(f"   コンテナ{i}: {counts[i]:3d}個  体積 {vols[i]:.3f}/{caps[i]:.3f} "
              f"({100*vols[i]/caps[i]:5.1f}%)  prioritized={c.is_prioritized} shelf={c.require_shelf}")
    env.close()
