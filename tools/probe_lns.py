import os,sys,json,time,io,contextlib
ROOT='/home/takato/comp/nedo'
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'agents','submit'))
from src.ground_handling.env import GroundHandlingEnv
import agent as am
src=open(os.path.join(ROOT,'agents','submit','agent.py')).read()
# instrument offer() to log every trial and every improvement
src=src.replace("""            def offer(key, order, plan):
                nonlocal best_key, best_order, best_plan
                if key > best_key:""","""            TRIALS=[]
            def offer(key, order, plan):
                nonlocal best_key, best_order, best_plan
                TRIALS.append((key, key > best_key))
                self._trials = TRIALS
                if key > best_key:""")
ns={'__name__':'agent_instr','__file__':os.path.join(ROOT,'agents','submit','agent.py')}
exec(compile(src, 'agent_instr.py','exec'), ns)
Agent=ns['Agent']; ns['OPTIMIZE_TIME_BUDGET']=150.0
scenes=json.load(open('scenes_offline.json'))
for nm in ['O00','O05','O15']:
    cfg=scenes[nm]
    with contextlib.redirect_stdout(io.StringIO()):
        env=GroundHandlingEnv(config=cfg,verbose=False); env.reset_settings()
        ag=Agent(module_path=''); ag.get_init_states(env.get_init_states())
        t0=time.perf_counter(); ag.optimize(env.get_info_for_optimization())
        el=time.perf_counter()-t0; env.close()
    tr=getattr(ag,'_trials',[])
    imp=[i for i,(k,b) in enumerate(tr) if b]
    prefixes=[k[0] for k,_ in tr]
    print(f'{nm}: {el:.0f}s, trials={len(tr)}, improvements at trial {imp}')
    print(f'    prefix per trial: first4={prefixes[:4]}  max={max(prefixes) if prefixes else 0}  '
          f'distinct={sorted(set(prefixes))[:12]}')
