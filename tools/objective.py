"""The quantity the leaderboard actually rewards.

Two facts fix this shape:

* LB is the 5-metric composite, not fill. LB 53.4 was observed against a
  local fill of 22.7-26.9 and a local composite of 57.3 (and, once the
  scene generator's distribution was corrected, 53.9-54.8).
* Below the README's "minimum items" threshold everything but fill scores
  zero. LB 27.27 at 42.3% of items placed matches fill; 53.4 at 46.5%
  matches the composite. So the reward is a step function with the cliff
  somewhere in 42-46%.

Optimising the composite alone is actively dangerous: it is maximised by
placing almost nothing (SUPPORT_MIN_COVER=1.0 scores 60.7 at 31.7%
placed, which on the real leaderboard collapses to fill's 13.6). Scoring
through the cliff instead makes "keep placing enough to stay over the
threshold" part of the objective rather than a side constraint.

The cliff's exact position is unknown, so the objective averages over
several plausible positions rather than betting on one.
"""

import math

METRICS = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']
THRESHOLDS = (40.0, 46.0, 50.0)

# Centre and width of the smoothed cliff used for *optimisation*.
#
# The hard step is the right model of the leaderboard but the wrong thing to
# optimise against. Measured on the 32-scene dev pool: only 6 scenes sit
# anywhere near the cliff (40-52% placed), and one scene crossing moves the
# objective by 0.67. The first CEM run gained +3.50 on its training pool --
# arithmetically, tipping about five of those six borderline scenes -- and
# then lost 0.87 on four unseen pools. It had not learned a better policy,
# it had memorised which training scenes were borderline.
#
# Replacing the step with a logistic ramp keeps the incentive ("stay well
# clear of the cliff") while removing the discontinuity that makes a single
# scene worth 0.67 and therefore worth overfitting to.
SMOOTH_CENTRE = 46.0
SMOOTH_WIDTH = 6.0


def composite(row):
    return sum(row.get(k, 0.0) for k in METRICS) / len(METRICS)


def scene_score(row, threshold):
    return composite(row) if row.get('pct', 0.0) >= threshold else row.get('fill', 0.0)


def est_lb(rows, threshold):
    if not rows:
        return 0.0
    return sum(scene_score(r, threshold) for r in rows) / len(rows)


def objective(rows, thresholds=THRESHOLDS):
    """Mean estimated leaderboard score, averaged over cliff positions.

    This is the *reporting* objective -- the honest model of the scoring
    rule. For search, prefer smooth_objective.
    """
    if not rows:
        return 0.0
    return sum(est_lb(rows, t) for t in thresholds) / len(thresholds)


def scene_score_smooth(row, centre=SMOOTH_CENTRE, width=SMOOTH_WIDTH):
    """Composite, faded into fill-only across the cliff instead of stepping."""
    c = composite(row)
    f = row.get('fill', 0.0)
    z = (row.get('pct', 0.0) - centre) / width
    t = 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
    return f + (c - f) * t


def smooth_objective(rows, centre=SMOOTH_CENTRE, width=SMOOTH_WIDTH):
    """The search objective. Same incentives, far less variance per scene."""
    if not rows:
        return 0.0
    return sum(scene_score_smooth(r, centre, width) for r in rows) / len(rows)


def summary(rows):
    n = max(len(rows), 1)
    out = {k: sum(r.get(k, 0.0) for r in rows) / n for k in METRICS}
    out['composite'] = sum(composite(r) for r in rows) / n
    out['packed'] = sum(r.get('pct', 0.0) for r in rows) / n
    out['objective'] = objective(rows)
    for t in THRESHOLDS:
        out[f'estlb{int(t)}'] = est_lb(rows, t)
        out[f'below{int(t)}'] = sum(1 for r in rows if r.get('pct', 0.0) < t)
    return out
