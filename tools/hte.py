"""Where does a treatment help, and how much of its noise is explainable?

Every A/B here is reported as one number -- "composite +1.68 (se 0.63),
38W/25L". That average hides two things worth having:

  1. How much of the scene-to-scene spread in the *effect* is structure
     rather than noise. Structure is removable: regress it out and the same
     64 scenes carry the power of many more.
  2. Which scenes the treatment helps. A treatment that is +4 on half the
     pool and -2 on the other half reports as +1 and gets shipped
     everywhere, when it should be shipped conditionally -- or reveals what
     it is really doing.

This fits Delta_i = effect on scene i against scene descriptors, by least
squares with a leave-one-out check so a flattering in-sample R^2 cannot be
mistaken for a real one.

    ../.venv/bin/python hte.py --base OFF_A --treat OFF_E --scenes scenes_off.json
"""
import argparse, json, math, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']


def comp(r):
    return sum(r.get(k, 0.0) for k in K) / 5.0


def load(tag):
    p = os.path.join(ROOT, 'tools', f'ab_{tag}.json')
    return {r['scene']: r for r in json.load(open(p))}


def scene_features(cfg, base_row):
    cl = cfg['containers']['container_list']
    items = cfg['item_stream']['item_list']
    vols = [i['length'] * i['width'] * i['height'] for i in items]
    return {
        'n_containers': float(len(cl)),
        'has_shelf': float(any(c.get('require_shelf') for c in cl)),
        'has_prio_container': float(any(c.get('is_prioritized') for c in cl)),
        'n_items': float(len(items)),
        'soft_frac': sum(1.0 for i in items if i.get('is_soft')) / len(items),
        'prio_frac': sum(1.0 for i in items if i.get('is_prioritized')) / len(items),
        'item_vol_cv': statistics.pstdev(vols) / statistics.mean(vols),
        'ceiling': float(base_row['ceiling']),
        # How full the baseline got: the headroom a treatment has to work with.
        'base_pct': float(base_row['pct']),
        'base_stability': float(base_row.get('stability_score', 0.0)),
    }


def ols(X, y):
    """Least squares with a tiny ridge, so a collinear column cannot blow up."""
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) + (1e-6 if a == b else 0.0)
            for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    # Gaussian elimination
    M = [row[:] + [Xty[a]] for a, row in enumerate(XtX)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(p):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [v - f * w for v, w in zip(M[r], M[c])]
    return [M[a][p] for a in range(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--treat', required=True)
    ap.add_argument('--scenes', default='scenes_off.json')
    a = ap.parse_args()

    cfgs = json.load(open(os.path.join(ROOT, 'tools', a.scenes)))
    b, t = load(a.base), load(a.treat)
    names = sorted(set(b) & set(t) & set(cfgs))
    d = [comp(t[s]) - comp(b[s]) for s in names]
    feats = [scene_features(cfgs[s], b[s]) for s in names]
    keys = sorted(feats[0])
    n = len(names)

    mean = statistics.mean(d)
    sd = statistics.pstdev(d)
    print(f"{a.base} -> {a.treat}   n={n}")
    print(f"raw effect  mean {mean:+.2f}  sd {sd:.2f}  se {sd / math.sqrt(n - 1):.2f}  "
          f"W/L {sum(1 for x in d if x > 0)}/{sum(1 for x in d if x < 0)}\n")

    # standardise so coefficients are comparable
    mu = {k: statistics.mean(f[k] for f in feats) for k in keys}
    sg = {k: (statistics.pstdev([f[k] for f in feats]) or 1.0) for k in keys}
    X = [[1.0] + [(f[k] - mu[k]) / sg[k] for k in keys] for f in feats]
    beta = ols(X, d)
    if beta is None:
        print('singular design')
        return
    fit = [sum(bb * xx for bb, xx in zip(beta, row)) for row in X]
    resid = [y - f for y, f in zip(d, fit)]
    ss_tot = sum((y - mean) ** 2 for y in d)
    r2 = 1 - sum(r * r for r in resid) / ss_tot

    # leave-one-out: the only R^2 worth quoting with p=11 and n=64
    loo_err = []
    for i in range(n):
        Xi = [X[j] for j in range(n) if j != i]
        yi = [d[j] for j in range(n) if j != i]
        bi = ols(Xi, yi)
        if bi is None:
            continue
        loo_err.append(d[i] - sum(bb * xx for bb, xx in zip(bi, X[i])))
    loo_r2 = 1 - sum(e * e for e in loo_err) / ss_tot

    print(f"effect model on {len(keys)} scene descriptors:")
    print(f"  in-sample R2 {r2:.3f}   leave-one-out R2 {loo_r2:.3f}")
    print(f"  residual sd  {statistics.pstdev(resid):.2f}  (raw {sd:.2f})")
    if loo_r2 > 0:
        print(f"  -> honest variance removed {100 * loo_r2:.0f}%, "
              f"equivalent to {1 / (1 - loo_r2):.2f}x the scenes")
    else:
        print("  -> nothing generalisable: the spread is noise or unmodelled by these features")

    se_b = statistics.pstdev(resid) / math.sqrt(max(n - len(keys) - 1, 1))
    print("\n  coefficient (composite points per 1 sd of the feature):")
    for k, bb in sorted(zip(keys, beta[1:]), key=lambda kv: -abs(kv[1])):
        star = '  *' if abs(bb) > 2 * se_b else ''
        print(f"    {k:20} {bb:+7.2f}{star}")
    print(f"  (approx se on each: {se_b:.2f}; * = |coef| > 2 se)")

    # Would conditional deployment beat blanket deployment?
    gain = [max(0.0, f) for f in fit]
    print(f"\n  blanket deployment   {mean:+.2f} per scene")
    print(f"  deploy where the model predicts a gain: "
          f"{sum(d[i] for i in range(n) if fit[i] > 0) / n:+.2f} per scene "
          f"({sum(1 for f in fit if f > 0)}/{n} scenes)")


if __name__ == '__main__':
    main()
