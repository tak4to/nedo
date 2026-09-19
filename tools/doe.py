"""Screening design over the constants that single A/Bs rejected.

Twenty-four settings were measured one at a time and rejected. Every one of
them was measured at se about 1.05 (32 scenes, one search seed), which is
17% power against a +1.0 effect -- so "rejected" and "not visible" were never
separated. Two things now say to go back:

  * tools/varcomp.py: the spread in these A/Bs is 100% search-seed noise,
    Var(b) = 0. Power has no floor; se falls as 1/sqrt(total runs).
  * The single largest improvement in the project came from optimising
    fourteen weights *jointly* (CEM full14) after the same fourteen had
    lost one at a time. Interactions are doing real work here.

A two-level fractional factorial answers both at once. Each main effect is a
contrast over all N runs rather than a difference of two, so its variance is
4s^2/N instead of 2s^2: at N=16 that is eight times smaller, i.e. 2.8x the
precision of the one-at-a-time tests that rejected these, for the same number
of runs. Resolution IV keeps main effects clear of two-factor interactions.

The all-minus corner is the shipped agent, so the design is centred on the
status quo and every effect reads as "what moving this factor does from here".

    ../.venv/bin/python doe.py --emit > doe_runs.sh && bash doe_runs.sh
    ../.venv/bin/python doe.py --analyse
"""
import argparse, itertools, json, math, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']

# name, low (= shipped), high
FACTORS = [
    ('W_LEVEL_MATCH',        0.0,     400.0),
    ('USE_UNDER_OVERHANG',   0.0,     1.0),
    ('SUPPORT_MIN_COVER',    0.62978, 0.72),
    ('SUPPORT_CENTROID_TOL', 0.35,    0.45),
    ('WALL_HEIGHT_TOL',      0.035,   0.050),
    ('SEAM_TOL',             0.03,    0.05),
    ('WASTE_CELL',           0.03,    0.02),
    ('TOP_TOL',              0.012,   0.020),
]
# 2^(8-4) resolution IV: E=BCD, F=ACD, G=ABC, H=ABD
GENERATORS = [(4, (1, 2, 3)), (5, (0, 2, 3)), (6, (0, 1, 2)), (7, (0, 1, 3))]


def design():
    rows = []
    for base in itertools.product((-1, 1), repeat=4):
        r = list(base) + [0] * 4
        for j, src in GENERATORS:
            s = 1
            for i in src:
                s *= base[i]
            r[j] = s
        rows.append(r)
    return rows


def emit(scenes_file, n_scenes, salts, jobs, budget):
    d = design()
    names = list(json.load(open(os.path.join(ROOT, 'tools', scenes_file))))[:n_scenes]
    only = ','.join(names)
    print('#!/bin/bash')
    print('cd "$(dirname "$0")"')
    print(f'# {len(d)} design points x {len(salts)} seeds x {n_scenes} scenes')
    for i, row in enumerate(d):
        sets = ' '.join(f'--set p:{FACTORS[j][0]}={FACTORS[j][1 if v < 0 else 2]}'
                        for j, v in enumerate(row))
        for s in salts:
            print(f'../.venv/bin/python ab.py --scenes {scenes_file} --only {only} '
                  f'--jobs {jobs} --tag DOE_r{i:02d}_s{s} --optimize-budget {budget} '
                  f'--policy-budget 5.0 {sets} --set p:OFFLINE_SEED_SALT={s} '
                  f'> ab_DOE_r{i:02d}_s{s}.log 2>&1')
        print(f'echo "design point {i + 1}/{len(d)} done"')
    print('echo DOE_DONE')


def comp(r):
    return sum(r.get(k, 0.0) for k in K) / 5.0


def analyse(salts):
    d = design()
    runs = {}
    for i in range(len(d)):
        for s in salts:
            p = os.path.join(ROOT, 'tools', f'ab_DOE_r{i:02d}_s{s}.json')
            if not os.path.exists(p):
                raise SystemExit(f'missing {p} -- run doe_runs.sh first')
            runs[(i, s)] = {r['scene']: comp(r) for r in json.load(open(p))}
    scenes = sorted(set.intersection(*[set(v) for v in runs.values()]))
    blocks = [(sc, s) for sc in scenes for s in salts]
    n_b = len(blocks)

    # Within each (scene, seed) block the design is balanced, so a main effect
    # is just the contrast of that block's 16 numbers -- the scene's own level
    # cancels exactly, the way pairing does in the A/B reports.
    print(f"{len(d)} design points x {len(salts)} seeds x {len(scenes)} scenes "
          f"= {len(d) * n_b} runs\n")
    base = statistics.mean(runs[(0, s)][sc] for sc in scenes for s in salts
                           if all(v < 0 for v in d[0]))
    all_minus = [i for i, row in enumerate(d) if all(v < 0 for v in row)]
    if all_minus:
        b = statistics.mean(runs[(all_minus[0], s)][sc] for sc in scenes for s in salts)
        print(f"shipped configuration (all factors low): composite {b:.2f}\n")

    print(f"{'factor':24} {'effect':>8} {'se':>6} {'t':>6}")
    results = []
    for j, (name, lo, hi) in enumerate(FACTORS):
        per_block = []
        for (sc, s) in blocks:
            plus = [runs[(i, s)][sc] for i in range(len(d)) if d[i][j] > 0]
            minus = [runs[(i, s)][sc] for i in range(len(d)) if d[i][j] < 0]
            per_block.append(statistics.mean(plus) - statistics.mean(minus))
        e = statistics.mean(per_block)
        se = statistics.pstdev(per_block) / math.sqrt(n_b - 1)
        results.append((abs(e / se) if se else 0.0, name, e, se, lo, hi))
    for t, name, e, se, lo, hi in sorted(results, reverse=True):
        star = ' *' if t > 2 else ''
        print(f"{name:24} {e:+8.2f} {se:6.2f} {t:6.2f}{star}   {lo} -> {hi}")
    print(f"\n  MDE at 80% power = 2.8 x se; the one-at-a-time tests that "
          f"rejected these ran at se 1.05 (MDE 2.9)")
    print("  * = |t| > 2. Resolution IV: main effects are clear of two-factor")
    print("  interactions, but two-factor interactions are aliased with each other.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emit', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n-scenes', type=int, default=32)
    ap.add_argument('--salts', default='0,1')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--budget', type=float, default=60.0)
    a = ap.parse_args()
    salts = [int(x) for x in a.salts.split(',')]
    if a.emit:
        emit(a.scenes, a.n_scenes, salts, a.jobs, a.budget)
    else:
        analyse(salts)


if __name__ == '__main__':
    main()
