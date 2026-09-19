"""How well does optimize()'s rollout predict the episode it is optimising?

Every offline improvement so far has run through the rollout: the search
ranks arrival orders by what the rollout says they are worth. Two changes
that gave the search *more* of that surrogate both lost (RRT -0.68;
the prefix cache -6.4 composite on the real task), which is the signature of
a biased predictor being over-exploited. So the question that decides what
to do next is not "how much better can the search get" but "how much of the
real outcome does the rollout actually explain".

For each offline scene this runs optimize(), records what the winning order
scored *inside* the rollout, then runs the real episode and measures what it
actually scored. It reports the correlation and the bias.

Reading:
  high r, small bias   -> the surrogate is fine; invest in the search
                          (beam search over the order, better operators)
  low r, or big bias   -> more search only overfits; invest in the rollout
                          (settle drift, scenario averaging, stability proxy)

    ../.venv/bin/python surrogate_fit.py --scenes scenes_off.json --jobs 12
"""
import argparse, contextlib, io, json, math, os, shutil, statistics, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']

PATCH = ("""            self._plan = best_plan""",
         """            self._opt_key = best_key
            self._plan = best_plan""")


def build(dst):
    shutil.copytree(os.path.join(ROOT, 'agents', 'submit'), dst,
                    ignore=shutil.ignore_patterns('__pycache__'))
    p = os.path.join(dst, 'agent.py')
    s = open(p).read()
    if PATCH[0] not in s:
        raise SystemExit('surrogate_fit: agent.py drifted, anchor missing')
    open(p, 'w').write(s.replace(*PATCH, 1))
    return dst


def _one(job):
    name, cfg, agent_dir, budget = job
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, agent_dir)
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    import scores
    A.OPTIMIZE_TIME_BUDGET = budget
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = A.Agent(module_path=agent_dir)
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        pred_n, pred_c = getattr(ag, '_opt_key', (0, 0.0))
        env.reset_item_stream()
        obs, _ = env.reset(seed=42)
        term = False
        steps = 0
        while not term and steps < 500:
            obs, _, term, _, _ = env.step(ag.policy(observation=obs))
            steps += 1
        rep = env.evaluate()
        truth = scores.all_scores(env)
        n_packed = sum(len(c.packed_items) for c in env.container_manager.containers)
        total = env.num_total_items
        row = dict(fill=rep['fill_score'], **{k: truth[k] for k in METRICS[1:]})
        env.close()
    return dict(scene=name, pred_n=pred_n, pred_comp=pred_c,
                real_n=n_packed, total=total,
                real_comp4=sum(row[k] for k in METRICS if k != 'stability_score') / 4.0,
                real_comp5=sum(row[k] for k in METRICS) / 5.0,
                **row)


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--budget', type=float, default=60.0)
    ap.add_argument('--out', default='surrogate_fit.json')
    args = ap.parse_args()

    scenes = json.load(open(os.path.join(ROOT, 'tools', args.scenes)))
    scenes = {k: v for k, v in scenes.items() if v['agent'].get('optimize')}
    tmp = tempfile.mkdtemp(prefix='surrfit_')
    try:
        agent_dir = build(os.path.join(tmp, 'submit'))
        jobs = [(n, c, agent_dir, args.budget) for n, c in scenes.items()]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_one, jobs))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rows.sort(key=lambda r: r['scene'])
    json.dump(rows, open(os.path.join(ROOT, 'tools', args.out), 'w'), indent=1)

    pn = [r['pred_n'] for r in rows]
    rn = [r['real_n'] for r in rows]
    pc = [r['pred_comp'] for r in rows]
    c4 = [r['real_comp4'] for r in rows]
    c5 = [r['real_comp5'] for r in rows]
    print(f"n = {len(rows)} offline scenes, optimize budget {args.budget:.0f}s\n")
    print(f"{'quantity':34} {'pred':>7} {'real':>7} {'bias':>7} {'r':>6}")
    print(f"{'items placed':34} {statistics.mean(pn):7.1f} {statistics.mean(rn):7.1f} "
          f"{statistics.mean(pn) - statistics.mean(rn):+7.1f} {pearson(pn, rn):6.3f}")
    print(f"{'composite (4 metrics, proxy scale)':34} {statistics.mean(pc):7.2f} "
          f"{statistics.mean(c4):7.2f} {statistics.mean(pc) - statistics.mean(c4):+7.2f} "
          f"{pearson(pc, c4):6.3f}")
    print(f"{'-> vs the scored 5-metric mean':34} {'':>7} {statistics.mean(c5):7.2f} "
          f"{'':>7} {pearson(pc, c5):6.3f}")
    resid = [p - r for p, r in zip(pc, c4)]
    print(f"\nresidual sd {statistics.pstdev(resid):.2f} "
          f"(real composite sd across scenes {statistics.pstdev(c4):.2f})")
    over = sum(1 for p, r in zip(pn, rn) if p > r)
    print(f"rollout over-counts items on {over}/{len(rows)} scenes, "
          f"under-counts on {sum(1 for p, r in zip(pn, rn) if p < r)}")


if __name__ == '__main__':
    main()
