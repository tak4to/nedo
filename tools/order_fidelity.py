"""Within a scene, does a better rollout score mean a better episode?

surrogate_fit.py answers the between-scene question (r = 0.98 on items
placed, 0.52 on composite), but the search never compares two scenes -- it
compares two arrival orders for the *same* scene. A per-scene offset in the
rollout is harmless for that; only a ranking error is not.

So: take one scene, generate several arrival orders, score each one twice --
once by optimize()'s own rollout, once by actually running the episode --
and correlate the two rankings inside the scene.

  high within-scene rank correlation -> the rollout ranks orders correctly,
      the search just is not finding the good ones. Invest in search
      (beam search over the order).
  low  -> the search is sorting noise. Invest in the rollout itself
      (settle drift, scenario averaging) before making it faster or wider.

    ../.venv/bin/python order_fidelity.py --scenes scenes_off.json --n 10 --orders 10
"""
import argparse, contextlib, io, json, os, random, shutil, statistics, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']

# Hand the rollout and its spec lookup out of optimize()'s closure so the
# probe can score an arbitrary order with the exact code the search uses.
PATCH = ("""            self._opt_key = best_key
            self._plan = best_plan""",
         """            self._opt_key = best_key
            self._rollout = rollout
            self._specs_of = specs_of
            self._plan = best_plan""")
BASE_PATCH = ("""            self._plan = best_plan""",
              """            self._opt_key = best_key
            self._plan = best_plan""")


def build(dst):
    shutil.copytree(os.path.join(ROOT, 'agents', 'submit'), dst,
                    ignore=shutil.ignore_patterns('__pycache__'))
    p = os.path.join(dst, 'agent.py')
    s = open(p).read()
    if BASE_PATCH[0] not in s:
        raise SystemExit('order_fidelity: agent.py drifted, anchor missing')
    s = s.replace(*BASE_PATCH, 1)
    s = s.replace(*PATCH, 1)
    open(p, 'w').write(s)
    return dst


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _one(job):
    name, cfg, agent_dir, n_orders, seed = job
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, agent_dir)
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    import scores
    A.OPTIMIZE_TIME_BUDGET = 8.0
    rng = random.Random(seed)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = A.Agent(module_path=agent_dir)
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        items = env.get_info_for_optimization()
        best = ag.optimize(items)
        env.close()
        rollout, specs_of = ag._rollout, ag._specs_of

        # A spread of orders around the incumbent, the way ruin-and-recreate
        # would reach them: a few random segment relocations each.
        orders = [list(best)]
        n = len(best)
        while len(orders) < n_orders:
            o = list(best)
            for _ in range(rng.randint(1, 3)):
                k = rng.randint(2, max(2, n // 6))
                i = rng.randrange(max(1, n - k))
                seg = o[i:i + k]
                del o[i:i + k]
                j = rng.randrange(len(o) + 1)
                o[j:j] = seg
            if len(o) == n and set(o) == set(best):
                orders.append(o)

        pred_n, pred_c, real_n, real_c = [], [], [], []
        for o in orders:
            key, _, _, _ = rollout(specs_of(o), time.perf_counter() + 30.0)
            pred_n.append(key[0]); pred_c.append(key[1])
            e = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
            e.reset_settings()
            ag2 = A.Agent(module_path=agent_dir)
            ag2.get_init_states(e.get_init_states())
            e.set_item_order(o)
            e.reset_item_stream()
            obs, _ = e.reset(seed=42)
            term, steps = False, 0
            while not term and steps < 500:
                obs, _, term, _, _ = e.step(ag2.policy(observation=obs))
                steps += 1
            rep = e.evaluate()
            truth = scores.all_scores(e)
            row = dict(fill=rep['fill_score'], **{k: truth[k] for k in METRICS[1:]})
            real_n.append(sum(len(c.packed_items) for c in e.container_manager.containers))
            real_c.append(sum(row[k] for k in METRICS if k != 'stability_score') / 4.0)
            e.close()
    return dict(scene=name, pred_n=pred_n, pred_c=pred_c, real_n=real_n, real_c=real_c,
                rho_n=spearman(pred_n, real_n), rho_c=spearman(pred_c, real_c),
                spread_pred=max(pred_c) - min(pred_c), spread_real=max(real_c) - min(real_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n', type=int, default=10, help='scenes to sample')
    ap.add_argument('--orders', type=int, default=10, help='orders per scene')
    ap.add_argument('--jobs', type=int, default=10)
    args = ap.parse_args()

    scenes = json.load(open(os.path.join(ROOT, 'tools', args.scenes)))
    scenes = {k: v for k, v in scenes.items() if v['agent'].get('optimize')}
    names = list(scenes)[:args.n]
    tmp = tempfile.mkdtemp(prefix='ordfid_')
    try:
        agent_dir = build(os.path.join(tmp, 'submit'))
        jobs = [(n, scenes[n], agent_dir, args.orders, 900 + i) for i, n in enumerate(names)]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_one, jobs))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(rows, open(os.path.join(ROOT, 'tools', 'order_fidelity.json'), 'w'), indent=1)

    print(f"{len(rows)} scenes x {args.orders} orders, within-scene Spearman\n")
    print(f"{'scene':8} {'rho(items)':>11} {'rho(comp)':>10} "
          f"{'pred spread':>12} {'real spread':>12}")
    for r in rows:
        print(f"{r['scene']:8} {r['rho_n']:11.2f} {r['rho_c']:10.2f} "
              f"{r['spread_pred']:12.2f} {r['spread_real']:12.2f}")
    print(f"\nmean rho(items) {statistics.mean(r['rho_n'] for r in rows):+.3f}   "
          f"mean rho(composite) {statistics.mean(r['rho_c'] for r in rows):+.3f}")
    print(f"mean spread: rollout says {statistics.mean(r['spread_pred'] for r in rows):.2f}, "
          f"reality {statistics.mean(r['spread_real'] for r in rows):.2f}")

    # --- what two-stage selection would be worth ----------------------------
    #
    # optimize() today returns argmax over the rollout's own score. Racing
    # (irace, successive halving) says: let the cheap evaluator shortlist and
    # let a more faithful one decide. The ceiling on that is what an oracle
    # would pick out of the same shortlist, so measure it directly.
    def stats(f):
        v = [f(r) for r in rows]
        return statistics.mean(v)

    def real_of_argmax(r, k):
        idx = sorted(range(len(r['pred_c'])), key=lambda i: -r['pred_c'][i])[:k]
        return max(r['real_c'][i] for i in idx)

    rand = stats(lambda r: statistics.mean(r['real_c']))
    cur = stats(lambda r: real_of_argmax(r, 1))
    best = stats(lambda r: max(r['real_c']))
    print("\nvalue of selection (mean true composite of the order returned):")
    print(f"  a random order from the pool            {rand:6.2f}")
    print(f"  what optimize returns today (argmax)    {cur:6.2f}   "
          f"({cur - rand:+.2f} over random)")
    for k in (2, 3, 5, 8):
        if k <= len(rows[0]['pred_c']):
            v = stats(lambda r, k=k: real_of_argmax(r, k))
            print(f"  oracle over the rollout's top {k:<2d}        {v:6.2f}   "
                  f"({v - cur:+.2f} over today)")
    print(f"  oracle over every order                 {best:6.2f}   "
          f"({best - cur:+.2f} over today)")
    print("\n  the top-k lines are the ceiling for a perfect second-stage judge;")
    print("  a real one with correlation r captures roughly a fraction of it.")


if __name__ == '__main__':
    main()
