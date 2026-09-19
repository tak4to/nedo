"""Split the noise in an A/B into its sources, and size the experiment from that.

Every offline A/B here is reported as one mean with a standard error built
from the scene-to-scene spread of the paired difference -- sd about 5, hence
"detecting +1.0 needs 200 scenes". docs/2026-09-06-戦略まとめ.md draws the
conclusion that this spread is treatment heterogeneity and therefore
irreducible: measuring the same thing again would not shrink it.

That conclusion is testable, and it decides how every future experiment
should be shaped. Write the paired difference on scene i under search seed j
as

    d_ij = tau + b_i + e_ij

with b_i the scene's own response to the treatment and e_ij the noise from
which local optimum the search happened to land in. Run several seeds on
several scenes and the two separate:

    Var(e) from the spread within a scene across seeds
    Var(b) from the spread of scene means, minus Var(e)/m

If Var(b) dominates, the spread really is heterogeneity: only more scenes
help, and it is worth asking which scenes. If Var(e) dominates, the spread
is the algorithm's own randomness: it averages away, and the same budget
buys far more power spent on seeds -- which also makes the handful of real
tasks usable as instruments instead of anecdotes.

    ../.venv/bin/python varcomp.py --base VC_b --treat VC_t --salts 0,1,2,3
"""
import argparse, json, math, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']


def comp(r):
    return sum(r.get(k, 0.0) for k in K) / 5.0


def load(tag):
    return {r['scene']: r for r in json.load(open(os.path.join(ROOT, 'tools', f'ab_{tag}.json')))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='tag prefix of the control arm')
    ap.add_argument('--treat', required=True)
    ap.add_argument('--salts', default='0,1,2,3')
    ap.add_argument('--metric', default='composite', choices=['composite', 'pct'])
    a = ap.parse_args()
    salts = [int(x) for x in a.salts.split(',')]
    val = comp if a.metric == 'composite' else (lambda r: r['pct'])

    B = {s: load(f'{a.base}_s{s}') for s in salts}
    T = {s: load(f'{a.treat}_s{s}') for s in salts}
    scenes = sorted(set.intersection(*[set(B[s]) for s in salts],
                                     *[set(T[s]) for s in salts]))
    m, n = len(salts), len(scenes)
    d = {sc: [val(T[s][sc]) - val(B[s][sc]) for s in salts] for sc in scenes}

    scene_means = [statistics.mean(d[sc]) for sc in scenes]
    grand = statistics.mean(scene_means)
    # within-scene (seed) variance, pooled
    var_e = statistics.mean(statistics.variance(d[sc]) for sc in scenes)
    # between-scene variance of the means, corrected for the seed noise in them
    var_means = statistics.variance(scene_means)
    var_b = max(0.0, var_means - var_e / m)

    print(f"{a.treat} vs {a.base} on {a.metric}: {n} scenes x {m} search seeds "
          f"= {n * m * 2} runs\n")
    print(f"  effect            {grand:+.2f}")
    print(f"  Var(e)  seed       {var_e:8.2f}   sd {math.sqrt(var_e):5.2f}   "
          f"<- same scene, different search seed")
    print(f"  Var(b)  scene      {var_b:8.2f}   sd {math.sqrt(var_b):5.2f}   "
          f"<- genuine treatment heterogeneity")
    tot = var_e + var_b
    print(f"  share of spread that is seed noise: {100 * var_e / tot:.0f}%\n")

    print("  standard error of the effect, by how the budget is spent:")
    print(f"    {'scenes':>8} {'seeds':>7} {'runs':>7} {'se':>7}")
    for ns, ms in ((n, 1), (n, m), (64, 1), (64, 2), (64, 4), (128, 1), (32, 8)):
        se = math.sqrt(var_b / ns + var_e / (ns * ms))
        print(f"    {ns:8d} {ms:7d} {2 * ns * ms:7d} {se:7.2f}")
    print("\n  minimum detectable effect (80% power, two-sided 5%) = 2.8 x se")

    # one real task, many seeds: is it usable as an instrument?
    se1 = math.sqrt(var_b + var_e / m)
    print(f"\n  a single scene at {m} seeds carries se {se1:.2f}; "
          f"to reach se 1.0 it needs {math.ceil(var_e / max(1.0 - var_b, 1e-9))} seeds"
          if var_b < 1.0 else
          f"\n  a single scene cannot reach se 1.0: heterogeneity alone is "
          f"sd {math.sqrt(var_b):.2f}")

    print("\n  per-scene effects (mean over seeds +- within-scene sd):")
    for sc in scenes:
        print(f"    {sc:6} {statistics.mean(d[sc]):+7.2f} +- {statistics.stdev(d[sc]):5.2f}"
              f"   [{' '.join(f'{x:+.1f}' for x in d[sc])}]")


if __name__ == '__main__':
    main()
