"""Pre-submission freeze check (docs/手荷物積付コンペ top10 到達戦略 §9).

Unzips a submission into a clean temp dir and runs it through the real
`EvaluationApp` (spawned worker, RLIMIT_AS = max_mem, the evaluator's own
timeouts), then reports whether the zip would survive on the judging box:

  static   layout (one dir holding agent.py + class Agent), dir-vs-zip sync,
           imports outside the judging image (numpy/pybullet/gymnasium/PIL/torch),
           unguarded third-party imports
  dynamic  every task finishes with an evaluation, no policy/optimize timeout,
           worst policy/optimize wall time vs the limit, peak child RSS vs max_mem

The judging box is 4 vCPU; the dev box is not. `--cpus 4 --load 2` pins the
whole run (children inherit the affinity) to 4 CPUs and adds busy-loop
processes on them, which is the cheap way to see how much wall-clock margin
the agent's own deadlines leave when the CPU is slower than ours.

    cd <repo root>
    .venv/bin/python tools/preflight.py --zip agents/submit.zip --cpus 4 --load 2
    .venv/bin/python tools/preflight.py --zip agents/submit.zip --static-only

Exit status 0 = PASS (warnings allowed), 1 = FAIL.
"""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# What dockerfiles/Dockerfile installs on top of the stdlib. numpy and
# pybullet arrive as gymnasium / pybullet dependencies, torch is CPU-only.
JUDGE_PKGS = {'numpy', 'pybullet', 'pybullet_utils', 'pybullet_data', 'gymnasium', 'PIL', 'torch'}
# Margins that make a pass mean something. The evaluator kills at the limit
# and substitutes a random action / the unsorted stream, so being at 99% of
# the limit on a slower box is a real risk, not a rounding error.
POLICY_MARGIN = 0.85     # worst policy call must stay under 85% of policy_timeout
OPTIMIZE_MARGIN = 0.95
MEM_MARGIN = 0.60        # peak RSS under 60% of max_mem (RLIMIT_AS counts virtual size too)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, what, detail=''):
        self.rows.append((level, what, detail))
        print(f'  [{level:4}] {what}' + (f' -- {detail}' if detail else ''))

    @property
    def failed(self):
        return any(r[0] == 'FAIL' for r in self.rows)


def _digest(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _imports(src):
    """(module, unguarded_at_top_level) for every import in a source file."""
    tree = ast.parse(src)
    out = []

    def walk(node, guarded, top):
        for ch in ast.iter_child_nodes(node):
            g = guarded or isinstance(ch, ast.Try)
            is_fn = isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(ch, ast.Import):
                out.extend((a.name.split('.')[0], top and not guarded) for a in ch.names)
            elif isinstance(ch, ast.ImportFrom) and ch.level == 0 and ch.module:
                out.append((ch.module.split('.')[0], top and not guarded))
            walk(ch, g, top and not is_fn and not isinstance(ch, ast.ClassDef))

    walk(tree, False, True)
    return out


def static_checks(zip_path, rep):
    print(f'\n== static: {zip_path}')
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            rep.add('FAIL', 'corrupt zip member', bad)
        names = [n for n in z.namelist() if not n.endswith('/')]
        tops = {n.split('/')[0] for n in names}
        agents = [n for n in names if os.path.basename(n) == 'agent.py']
        if len(agents) != 1:
            rep.add('FAIL', 'need exactly one agent.py', str(agents))
            return None
        root = os.path.dirname(agents[0])
        rep.add('ok', 'layout', f'agent.py at {agents[0]!r}, top-level entries {sorted(tops)}')
        if len(tops) != 1:
            rep.add('WARN', 'several top-level entries; sample_submit.zip has a single dir')
        pycache = [n for n in names if '__pycache__' in n or n.endswith('.pyc')]
        if pycache:
            rep.add('WARN', 'bytecode inside the zip', f'{len(pycache)} files')

        # dir-vs-zip sync: the zip is what gets judged, the dir is what gets A/B'd
        src_dir = os.path.join(os.path.dirname(os.path.abspath(zip_path)), root or '.')
        if root and os.path.isdir(src_dir):
            drift = []
            for n in names:
                p = os.path.join(os.path.dirname(os.path.abspath(zip_path)), n)
                if not os.path.exists(p):
                    drift.append(f'{n}: missing on disk')
                elif hashlib.sha256(z.read(n)).hexdigest()[:12] != _digest(p):
                    drift.append(f'{n}: differs from working tree')
            if drift:
                rep.add('WARN', 'zip is not the working tree', '; '.join(drift))
            else:
                rep.add('ok', 'zip == working tree', src_dir)

        local = {os.path.splitext(os.path.basename(n))[0] for n in names if n.endswith('.py')}
        for n in names:
            if not n.endswith('.py'):
                continue
            for mod, unguarded in _imports(z.read(n).decode('utf-8')):
                if mod in sys.stdlib_module_names or mod in local:
                    continue
                if mod not in JUDGE_PKGS:
                    rep.add('FAIL', f'{n}: import {mod!r} is not in the judging image',
                            'add it to requirements.txt or guard it')
                elif unguarded and mod not in {'numpy'}:
                    rep.add('WARN', f'{n}: unguarded top-level import {mod!r}',
                            'wrap in try/except so a missing wheel cannot kill the agent')
        if 'requirements.txt' in {os.path.basename(n) for n in names}:
            rep.add('WARN', 'requirements.txt present',
                    'it triggers a pip install on the judge; make sure nothing in it can fail to resolve')
        return root


def _burn():
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def dynamic_checks(zip_path, config_path, tasks, cpus, load, rep):
    print(f'\n== dynamic: {config_path} tasks={tasks or "all"} cpus={cpus or "all"} load={load}')
    if cpus:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:cpus]))
    tmp = tempfile.mkdtemp(prefix='preflight_')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    agent_dir = os.path.dirname(next(
        os.path.join(tmp, n) for n in zipfile.ZipFile(zip_path).namelist()
        if os.path.basename(n) == 'agent.py'))
    # the worker is spawned; give it the extracted dir, not the repo's agents/
    os.environ['PYTHONPATH'] = os.pathsep.join([agent_dir, REPO] + (
        [os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # spawn hands the parent's sys.path to the worker (PYTHONPATH alone is ignored)
    sys.path[:0] = [agent_dir, REPO]

    from src.ground_handling import runner as runner_mod
    from src.ground_handling.app import EvaluationApp

    calls = []
    orig = runner_mod.TimedAgentRunner.call

    def timed(self, method_name, time_out_sec, *a, **kw):
        res = orig(self, method_name, time_out_sec, *a, **kw)
        calls.append((method_name, res[1], time_out_sec))
        return res

    runner_mod.TimedAgentRunner.call = timed
    import src.ground_handling.app as app_mod
    app_mod.TimedAgentRunner = runner_mod.TimedAgentRunner

    with open(config_path) as f:
        cfg = json.load(f)
    if tasks:
        cfg = {k: v for k, v in cfg.items() if k in tasks}
    cfg_path = os.path.join(tmp, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)

    burners = [mp.get_context('spawn').Process(target=_burn, daemon=True) for _ in range(load)]
    for b in burners:
        b.start()
    t0 = time.time()
    try:
        EvaluationApp(config_path=cfg_path, module_path=agent_dir + '/', agent_module='agent',
                      agent_class='Agent', result_dir=tmp, result_fname='res.json').run(verbose=False)
    finally:
        for b in burners:
            b.kill()
    wall = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024

    with open(os.path.join(tmp, 'res.json')) as f:
        results = json.load(f)
    for tid, r in results.items():
        ev = r.get('evaluation')
        if r.get('status') != 'success' or ev is None:
            rep.add('FAIL', f'task {tid} did not finish', (r.get('message') or '')[-300:])
        else:
            rep.add('ok', f'task {tid}', f"fill={ev.get('fill_score'):.2f} placed={ev.get('num_placed_items'):.3f}")
    missing = set(cfg) - set(results)
    if missing:
        rep.add('FAIL', 'tasks never ran', str(sorted(missing)))

    for method in ('get_init_states', 'optimize', 'policy'):
        rows = [(e, lim) for m, e, lim in calls if m == method]
        if not rows:
            continue
        worst, lim = max(rows)
        n_to = sum(1 for e, l in rows if e >= l * 0.999)
        frac = worst / lim
        margin = {'policy': POLICY_MARGIN, 'optimize': OPTIMIZE_MARGIN}.get(method, 0.9)
        lvl = 'FAIL' if n_to else ('WARN' if frac > margin else 'ok')
        rep.add(lvl, f'{method}: {len(rows)} calls, {n_to} timeouts',
                f'worst {worst:.2f}s of {lim:.0f}s ({100 * frac:.0f}%)')

    mem_lim = max((v['agent'].get('max_mem', 4) for v in cfg.values()), default=4) * 1024
    lvl = 'WARN' if peak_mb > MEM_MARGIN * mem_lim else 'ok'
    rep.add(lvl, 'peak child RSS', f'{peak_mb:.0f} MB of {mem_lim:.0f} MB max_mem')
    rep.add('ok', 'wall time', f'{wall:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=os.path.join(REPO, 'agents/submit.zip'))
    ap.add_argument('--config', default=os.path.join(REPO, 'configs/sample_config.json'))
    ap.add_argument('--tasks', default='', help='comma-separated task ids (default: all)')
    ap.add_argument('--cpus', type=int, default=0, help='pin to the first N CPUs (judge: 4)')
    ap.add_argument('--load', type=int, default=0, help='busy-loop processes competing for those CPUs')
    ap.add_argument('--static-only', action='store_true')
    args = ap.parse_args()

    rep = Report()
    root = static_checks(args.zip, rep)
    if root is not None and not args.static_only and not rep.failed:
        dynamic_checks(args.zip, args.config, set(filter(None, args.tasks.split(','))),
                       args.cpus, args.load, rep)
    lv = [r[0] for r in rep.rows]
    print(f"\nPREFLIGHT {'FAIL' if 'FAIL' in lv else 'PASS'}  ({lv.count('WARN')} warnings)")
    sys.exit(1 if 'FAIL' in lv else 0)


if __name__ == '__main__':
    main()
