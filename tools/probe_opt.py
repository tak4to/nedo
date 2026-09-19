"""Instrument optimize() and report what the offline search actually does.

Copies agents/submit into a temp dir, patches counters into optimize(), and
runs one scene's optimize() at the full budget. Reports:

  * rollouts issued vs. moves accepted, and when the last improvement landed
  * the distinct values the objective's primary term (prefix length) takes,
    i.e. how big the plateau is
  * the longest common prefix between each trial order and the incumbent,
    i.e. how much of every rollout is a re-simulation (upper bound on what a
    prefix cache could save)
  * the spread of the cog proxy across orders that tie on prefix length,
    i.e. what the current tiebreaker (prefix_volume) throws away

Measured on R000 (2026-09-08): 820 rollouts, 4 accepted, last improvement at
t=9.7s, and a 4.60-point cog spread among the 214 orders tied at prefix 30.
See docs/2026-09-08-オフライン最適化戦略.md.

    ../.venv/bin/python probe_opt.py                 # R000
    ../.venv/bin/python probe_opt.py --scenes scenes_off.json --only P00
"""
import argparse, collections, contextlib, io, json, os, shutil, statistics, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCHES = [
    ("OFFLINE_SEARCH = 'lns'",
     """OFFLINE_SEARCH = 'lns'

_STATS = {'rollouts': 0, 'trials': 0, 'accepts': 0, 'lcp': [], 'cut': [],
          'keys': [], 't0': 0.0, 'prefix': [], 'cog': [], 'walks': 0, 'prof': []}


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n"""),
    ("""        deadline = time.perf_counter() + OPTIMIZE_TIME_BUDGET
        try:""",
     """        deadline = time.perf_counter() + OPTIMIZE_TIME_BUDGET
        _STATS['t0'] = time.perf_counter()
        try:"""),
    ("""                allowed_misses = max(1, int(self._lookahead_k or 1))""",
     """                allowed_misses = max(1, int(self._lookahead_k or 1))
                _STATS['rollouts'] += 1
                _prof = []
                _t_roll = time.perf_counter()"""),
    ("""                for pos_i, spec in enumerate(trial_specs):
                    if time.perf_counter() > budget_deadline:""",
     """                for pos_i, spec in enumerate(trial_specs):
                    _prof.append(time.perf_counter() - _t_roll)
                    if time.perf_counter() > budget_deadline:"""),
    ("""                tail = ([s['index'] for s in trial_specs[stopped_at:]]""",
     """                _tm = 0.0; _num = 0.0; _H = 0.0
                for _cs in states.values():
                    _H = max(_H, _cs.geom['height'])
                    for _b in _cs.boxes:
                        if _b.get('is_static'):
                            continue
                        _tm += _b['mass']; _num += _b['mass'] * _b['center'][2]
                _prof.append(time.perf_counter() - _t_roll)
                _STATS['prof'].append(_prof)
                _STATS['prefix'].append(len(ordered))
                _STATS['cog'].append(
                    round(100.0 * (1.0 - (_num / _tm) / _H), 3) if (_tm and _H) else 0.0)
                tail = ([s['index'] for s in trial_specs[stopped_at:]]"""),
    ("""            def run_trial(trial_specs):
                nonlocal avg_trial
                t0 = time.perf_counter()
                key, order, plan = rollout(trial_specs, deadline)
                trial_times.append(time.perf_counter() - t0)
                avg_trial = sum(trial_times) / len(trial_times)
                return offer(key, order, plan)""",
     """            def run_trial(trial_specs):
                nonlocal avg_trial
                _STATS['trials'] += 1
                _STATS['lcp'].append(_lcp([s['index'] for s in trial_specs], cur_order))
                _STATS['cut'].append(cur_key[0])
                t0 = time.perf_counter()
                key, order, plan = rollout(trial_specs, deadline)
                dt = time.perf_counter() - t0
                trial_times.append(dt)
                avg_trial = sum(trial_times) / len(trial_times)
                acc = offer(key, order, plan)
                if acc:
                    _STATS['accepts'] += 1
                _STATS['keys'].append((round(time.perf_counter() - _STATS['t0'], 2),
                                       key[0], round(key[1], 4), round(dt, 3), acc))
                return acc"""),
    ("""                if merit(key) >= merit(best_key) - packer.RRT_DEV:
                    cur_key, cur_order = key, order""",
     """                if merit(key) >= merit(best_key) - packer.RRT_DEV:
                    cur_key, cur_order = key, order
                    _STATS['walks'] += 1"""),
]


def build_instrumented(dst):
    shutil.copytree(os.path.join(ROOT, 'agents', 'submit'), dst,
                    ignore=shutil.ignore_patterns('__pycache__'))
    p = os.path.join(dst, 'agent.py')
    s = open(p).read()
    for old, new in PATCHES:
        if old not in s:
            raise SystemExit(f'probe_opt: agent.py has drifted, patch anchor not found:\n{old[:80]}')
        s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_real.json')
    ap.add_argument('--only', default='R000')
    ap.add_argument('--budget', type=float, default=None, help='override OPTIMIZE_TIME_BUDGET')
    ap.add_argument('--set', action='append', default=[],
                    help='packer constant override, e.g. --set OFFLINE_TIEBREAK=0')
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, 'tools', args.scenes)))[args.only]
    if not cfg['agent'].get('optimize'):
        raise SystemExit(f'{args.only} has optimize=False -- optimize() is never called on it')

    tmp = tempfile.mkdtemp(prefix='probe_opt_')
    try:
        agent_dir = build_instrumented(os.path.join(tmp, 'submit'))
        sys.path.insert(0, os.path.join(ROOT, 'src'))
        sys.path.insert(0, agent_dir)
        from ground_handling.env import GroundHandlingEnv
        import agent as A
        import packer
        if args.budget is not None:
            A.OPTIMIZE_TIME_BUDGET = args.budget
        for kv in args.set:
            k, v = kv.split('=', 1)
            setattr(packer, k, float(v))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
            ag = A.Agent(module_path=agent_dir)
            env.reset_settings()
            ag.get_init_states(env.get_init_states())
            t = time.perf_counter()
            ag.optimize(env.get_info_for_optimization())
            wall = time.perf_counter() - t
            env.close()
        st = A._STATS
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(cfg['item_stream']['item_list'])
    print(f"scene {args.only}  optimize=True k={cfg['item_stream']['look_ahead']} n={n}")
    print(f"wall {wall:.1f}s  rollouts {st['rollouts']}  trials {st['trials']}  "
          f"record improvements {st['accepts']}  walk moves {st['walks']}")
    pre = st['prefix']
    print(f"prefix len: best {max(pre)}  mean {statistics.mean(pre):.1f}  "
          f"distinct {sorted(set(pre))}   <- plateau width")
    lcp = st['lcp']
    if lcp:
        print(f"LCP with incumbent: mean {statistics.mean(lcp):.1f} / {n}  "
              f"median {statistics.median(lcp)}  max {max(lcp)}   <- prefix-cache upside")
    tt = [k[3] for k in st['keys']]
    if tt:
        print(f"trial time: mean {statistics.mean(tt):.3f}s  "
              f"min {min(tt):.3f}  max {max(tt):.3f}")
    acc = [k for k in st['keys'] if k[4]]
    print(f"record improved at t = {[k[0] for k in acc]}   "
          f"(then {wall - (acc[-1][0] if acc else 0):.0f}s without a new record)")
    print(f"final key: items {st['keys'][-1][1] if st['keys'] else '-'}, "
          f"best tiebreak {max((k[2] for k in st['keys']), default=0):.3f}")

    by = collections.defaultdict(list)
    for pl, cg in zip(pre, st['cog']):
        by[pl].append(cg)
    print("\nprefix -> cog proxy across tied orders:")
    for pl in sorted(by)[-8:]:
        v = by[pl]
        print(f"  prefix {pl:3d}  n={len(v):4d}  cog {min(v):6.2f} / "
              f"{statistics.mean(v):6.2f} / {max(v):6.2f}   spread {max(v) - min(v):5.2f}")
    prof, lcps = st['prof'], st['lcp']
    if prof and lcps:
        share = []
        for pr, L in zip(prof[-len(lcps):], lcps):
            if len(pr) < 2 or pr[-1] <= 0:
                continue
            share.append(pr[min(L, len(pr) - 1)] / pr[-1])
        if share:
            avg = statistics.mean(share)
            print(f"\nprefix-cache gate: the shared prefix is "
                  f"{statistics.mean(lcps) / n:.0%} of the items but only "
                  f"{avg:.0%} of a rollout's time  ->  best case speedup "
                  f"{1 / max(1 - avg, 1e-6):.2f}x")
            print("  (best_placement is cheap while there is room and expensive once "
                  "there is not, so caching the prefix buys the cheap half)")

    b = max(by)
    import packer as _pk
    which = 'composite' if _pk.OFFLINE_TIEBREAK else 'volume'
    print(f"\nAt the incumbent prefix ({b}), tiebreak={which}: the first order the search "
          f"found there has cog {by[b][0]:.2f}, the best it visited has {max(by[b]):.2f} "
          f"(spread {max(by[b]) - by[b][0]:+.2f})")


if __name__ == '__main__':
    main()
