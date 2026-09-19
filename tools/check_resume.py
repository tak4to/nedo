"""Prove that resuming a rollout from a cached prefix changes nothing.

optimize() reuses the simulated prefix a trial shares with the order it was
derived from (agent.rollout's `resume`). That is only a budget optimisation
if it is exactly semantics-preserving, so this runs every resumed rollout a
second time from scratch and compares (key, order, plan) byte for byte --
the same standard the _landing speedup was held to.

    ../.venv/bin/python check_resume.py                      # R000
    ../.venv/bin/python check_resume.py --scenes scenes_off.json --only O00,O01
"""
import argparse, collections, contextlib, io, json, os, shutil, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCHES = [
    ("OFFLINE_SEARCH = 'lns'",
     """OFFLINE_SEARCH = 'lns'

_CHK = {'n': 0, 'bad': 0, 'at': [], 'ex': []}"""),
    ("""                t0 = time.perf_counter()
                key, order, plan, trail = rollout(trial_specs, deadline, resume_for(t_order))""",
     """                t0 = time.perf_counter()
                _r = resume_for(t_order)
                key, order, plan, trail = rollout(trial_specs, deadline, _r)
                if _r is not None:
                    _k2, _o2, _p2, _ = rollout(trial_specs, deadline, None)
                    _CHK['n'] += 1
                    _CHK['at'].append(_r[0])
                    if (key, order, plan) != (_k2, _o2, _p2):
                        _CHK['bad'] += 1
                        if len(_CHK['ex']) < 3:
                            _CHK['ex'].append((_r[0], key, _k2))"""),
]


def build(dst):
    shutil.copytree(os.path.join(ROOT, 'agents', 'submit'), dst,
                    ignore=shutil.ignore_patterns('__pycache__'))
    p = os.path.join(dst, 'agent.py')
    s = open(p).read()
    for old, new in PATCHES:
        if old not in s:
            raise SystemExit(f'check_resume: agent.py drifted, anchor missing:\n{old[:90]}')
        s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_real.json')
    ap.add_argument('--only', default='R000')
    ap.add_argument('--budget', type=float, default=30.0)
    args = ap.parse_args()

    scenes = json.load(open(os.path.join(ROOT, 'tools', args.scenes)))
    names = [n for n in args.only.split(',') if scenes.get(n, {}).get('agent', {}).get('optimize')]
    if not names:
        raise SystemExit('no offline (optimize=True) scenes selected')

    tmp = tempfile.mkdtemp(prefix='check_resume_')
    total = collections.Counter()
    ats = []
    try:
        agent_dir = build(os.path.join(tmp, 'submit'))
        sys.path.insert(0, os.path.join(ROOT, 'src'))
        sys.path.insert(0, agent_dir)
        from ground_handling.env import GroundHandlingEnv
        import agent as A
        import packer
        A.OPTIMIZE_TIME_BUDGET = args.budget
        # Force the cache on regardless of its shipped default, or this test
        # passes vacuously by never resuming anything.
        packer.OFFLINE_PREFIX_CACHE = 1.0
        for name in names:
            A._CHK.update(n=0, bad=0, at=[], ex=[])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                env = GroundHandlingEnv(config=scenes[name], verbose=False, render_mode=None)
                ag = A.Agent(module_path=agent_dir)
                env.reset_settings()
                ag.get_init_states(env.get_init_states())
                ag.optimize(env.get_info_for_optimization())
                env.close()
            c = A._CHK
            ats += c['at']
            total['n'] += c['n']; total['bad'] += c['bad']
            flag = 'OK' if c['bad'] == 0 else f"MISMATCH x{c['bad']}"
            print(f"{name:8} resumed {c['n']:5d} rollouts   {flag}")
            for ex in c['ex']:
                print(f"    at {ex[0]}: resumed {ex[1]} vs fresh {ex[2]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{total['n']} resumed rollouts, {total['bad']} differing from a fresh run")
    if ats:
        import statistics
        print(f"resume position: mean {statistics.mean(ats):.1f}  "
              f"median {statistics.median(ats)}  max {max(ats)}")
    sys.exit(1 if total['bad'] else 0)


if __name__ == '__main__':
    main()
