"""Ground-handling baggage loading agent.

Architecture (see docs/STRATEGY.md at the repo root for the full writeup):

  1. geometry.py  - a pure-math oracle that exactly mirrors the
     evaluator's own inclusion / transport-path checks, so we only ever
     propose actions we already know are physically valid. This is what
     lets us never fail an episode early.
  2. packer.py    - candidate generation (extreme points plus a grid
     fallback, with each candidate's resting height derived from where
     the item would land) and scoring (pack low and towards the back,
     lay items flat, keep the centre of mass low, route priority
     baggage to its own container).
  3. agent.py (this file) - wires the above into the required
     __init__/get_init_states/optimize/policy interface, with hard time
     budgets and defensive fallbacks so a slow or buggy corner case can
     never throw (an uncaught exception fails the *whole* task, whereas
     a timeout only costs one random step).

`policy()` is intentionally self-sufficient: it rebuilds all of its state
from the `observation` dict every call instead of trusting `self.*` set
by `get_init_states`, because a policy-timeout restarts the agent process
and the fresh process never receives `get_init_states` again.
"""
import os
import random
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np  # noqa: E402

import geometry  # noqa: E402
import packer  # noqa: E402
from packer import ContainerState, best_placement, choose_action  # noqa: E402

# How optimize() spends its 180s. Kept as an explicit switch because the
# three modes are genuinely different bets, measured over 40 offline scenes
# at the full budget (docs/GAMEPLAN.md #3.2, tools/ab.py):
#
#   'fixed' : the four fixed sort keys and nothing else. norm 31.72, and it
#             finishes in ~40s, leaving most of the budget unused.
#   'grasp' : plus up to 200 randomized sort keys. norm 31.72 -- byte for
#             byte the same answer as 'fixed' on every scene, for 135 extra
#             seconds. Randomly reweighting four item features cannot leave
#             the sliver of n! that the fixed keys already cover, which is
#             also why docs/STRATEGY.md #12.6 measured multi-start at zero.
#   'lns'   : plus ruin-and-recreate on the incumbent order, which moves in
#             permutation space directly. norm 33.43, +1.71 paired, better
#             on 23 scenes and worse on 8 (sign test p = 0.011).
#
# 'lns' wins on the pool but is the higher-variance bet: it optimises the
# rollout's prefix length, and the rollout is drift-free while the real run
# is not, so a gain against the surrogate does not always transfer (on
# sample task 000 it costs 6 points). That is the attacking half of the
# two-submission portfolio in docs/GAMEPLAN.md #5.2; 'fixed' is the
# defensive half, and agents/submit_safe.zip ships it.
OFFLINE_SEARCH = 'lns'

# Physics-twin settle gate: BUILT, CALIBRATED, MEASURED, AND TURNED OFF.
#
# twin.py replicates the evaluator's third gate exactly enough to flag 12
# of 12 episodes that really died on it, and the budget to run it was
# free -- once choose_action stopped starving (MIN_CALL_BUDGET), extra
# search time was worth only +0.07 norm, so ~0.5s of settle simulation
# costs nothing we were using. It still does not pay:
#
#   as a veto on the primary search   dev +0.45  holdout -0.77
#   as a preference (demote, never    dev +0.30  holdout -0.35
#     drop) plus the fallback ladder
#
# and the terminal-status histogram is unmoved (21 "nothing fits" / 11
# topple, before and after, on both pools). The reading that fits: by the
# time the best available placement is one that topples, the container is
# already finished, and refusing it only changes which of the two ways
# the episode ends. Toppling is a symptom of the jam, not its cause --
# which is why docs/GAMEPLAN.md #1 had this as the top priority for days
# and why it is now closed rather than merely untried.
#
# Kept behind this flag, with the plumbing intact (packer's `verify`
# hook, twin.py), because the twin is a correct and reusable oracle: an
# offline placement-level search can afford far more of it per decision
# than policy() can, which is a different bet from this one.
USE_SETTLE_TWIN = False
MAX_SETTLE_CHECKS = 12

POLICY_TIME_BUDGET = 5.0
OPTIMIZE_TIME_BUDGET = 150.0
INIT_TIME_BUDGET = 8.0


def _item_spec(d, pool_index=None):
    return {
        'index': d.get('index'),
        'pool_index': pool_index,
        'length': float(d['length']), 'width': float(d['width']), 'height': float(d['height']),
        'mass': float(d.get('mass', 1.0)),
        'is_prioritized': bool(d.get('is_prioritized', False)),
        'is_soft': bool(d.get('is_soft', False)),
    }


def _tiebreak(states):
    """Rank two arrival orders that placed the same number of items.

    Which is nearly all of them: optimize()'s primary key is a count, so
    over a 41! space the objective is mostly plateau, and this function
    decides where inside it the search settles.

    packer.pack_metrics scores the terminal state on four of the five
    metrics the leaderboard pays for. The alternative branch is what the
    search used until 2026-09-08 -- total volume placed -- kept switchable
    so the change can be A/B'd with `ab.py --set p:OFFLINE_TIEBREAK=0`.
    Volume is not a neutral tiebreak: it prefers the big boxes going down
    first, which stacks higher, which costs cog.
    """
    if packer.OFFLINE_TIEBREAK:
        return packer.pack_metrics(states)
    return sum(8.0 * b['half'][0] * b['half'][1] * b['half'][2]
               for cs in states.values() for b in cs.boxes)


def _fallback_action(observation, deadline=None, twins=None, pool_list_raw=None):
    """Last-resort ladder when the primary search finds nothing.

    Worth being exhaustive here, because the alternative is strictly worse
    and the downside of trying is exactly zero. When the search gives up we
    return a position we know is invalid, env.step fails it, and the episode
    ends on the state as it stands. If instead we return a *risky* placement
    -- one the oracle accepts as reachable but whose support we would
    normally refuse -- there are only two outcomes: the physics settles it
    and we carry on, or validator.place_item finds it displaced past the
    threshold, removes the item (leaving the container exactly as it was)
    and ends the episode. The failure case is byte-for-byte the state we
    would have ended on anyway, so a risky attempt is a free option.

    Ladder, each rung strictly weaker than the last:
      1. relax the priority/soft stacking constraint (a placement-score
         penalty is far cheaper than ending the episode),
      2. also drop the static-stability requirement entirely, accepting
         overhangs the settle step may or may not tolerate,
      3. only then, an unvalidated position -- at which point the episode is
         over regardless of what we return.
    Never allowed to raise.
    """
    if deadline is None:
        deadline = time.perf_counter() + 1.5
    try:
        pool_list = observation.get('pool_list') or []
        container_list = observation.get('container_list') or []
        if not pool_list or not container_list:
            return {
                'item_idx': 0, 'container_idx': 0,
                'place_pos': np.array([0.0, 0.0, 0.5], dtype=np.float32),
                'orientation': 0,
            }

        container_states = {c['index']: ContainerState(c) for c in container_list}
        pool = [(i, _item_spec(d, i)) for i, d in enumerate(pool_list)]

        # Both rungs get the *whole* remaining budget, not half each. Rung 2
        # only runs at all if rung 1 came back empty, and choose_action
        # returns as soon as it finds something, so giving rung 1 less time
        # than the old single attempt had is a pure regression -- measured as
        # one scene dropping from 18 placed items to 16. If rung 1 burns the
        # entire budget, rung 2 gets nothing and we land exactly where the old
        # code did; anything it finds in leftover time is free.
        verifier = None
        if twins and pool_list_raw:
            checks = [0]

            def verifier(cidx, spec):          # noqa: F811
                t = twins.get(cidx)
                if t is None or spec.get('pool_index') is None:
                    return None
                raw = pool_list_raw[spec['pool_index']]

                def verify(pos, orn):
                    if checks[0] >= 6 or time.perf_counter() > deadline:
                        return True
                    checks[0] += 1
                    try:
                        return t.settles(raw, pos, orn)
                    except Exception:
                        return True
                return verify

        for require_support in (True, False):
            action = choose_action(container_states, pool, deadline, relax=True,
                                   require_support=require_support, verifier=verifier)
            if action is not None:
                pool_idx, cidx, pos, orn = action
                return {
                    'item_idx': int(pool_idx), 'container_idx': int(cidx),
                    'place_pos': np.array(pos, dtype=np.float32), 'orientation': int(orn),
                }

        # Nothing fits anywhere we can find: syntactically valid, but not
        # guaranteed safe. Only reachable when the container is genuinely
        # full, in which case the episode is over either way.
        cidx0 = container_list[0]['index']
        g = container_states[cidx0].geom
        spec0 = pool[0][1]
        _, _, hz = geometry.half_extents(spec0['length'], spec0['width'], spec0['height'], 0)
        pos = ((g['x_lo'] + g['x_hi']) / 2.0, (g['y_lo'] + g['y_hi']) / 2.0,
               g['floor_struct_z'] + geometry.FLOOR_LIFT + hz)
        return {
            'item_idx': 0, 'container_idx': int(cidx0),
            'place_pos': np.array(pos, dtype=np.float32),
            'orientation': 0,
        }
    except Exception:
        traceback.print_exc()
        return {
            'item_idx': 0, 'container_idx': 0,
            'place_pos': np.array([0.0, 0.0, 0.5], dtype=np.float32),
            'orientation': 0,
        }


class Agent:
    def __init__(self, module_path: str):
        self.module_path = module_path
        self._known_containers = None
        self._lookahead_k = None
        # Best offline rollout's placements, kept for diagnostics and as
        # the hook a placement-level offline search would need.
        #
        # Deliberately NOT replayed by policy(). Measured over the 32-scene
        # pool: replaying the planned (container, x, y, orn) and recomputing
        # only the resting height scored 24.80 vs 25.72 for plain greedy,
        # and 38.1 vs 46.7 on the offline subset. The reason is structural
        # rather than a tuning problem -- the plan is produced by the *same*
        # greedy, but against design coordinates in a simulated container,
        # whereas policy() runs that greedy against the boxes as they
        # actually settled. Replaying therefore always trades a
        # better-informed argmax for a worse-informed one. A plan is only
        # worth replaying once the offline search chooses placements the
        # greedy would not (beam search over placements, docs/GAMEPLAN.md
        # #3.3c); optimising the *order*, which is what optimize() actually
        # returns, needs no replay at all.
        self._plan = {}        # item index -> (container_idx, cx, cy, orn)
        self._plan_rank = {}   # item index -> position in the planned order

    def get_init_states(self, init_states: dict):
        try:
            self._known_containers = init_states.get('container_list', [])
            self._lookahead_k = init_states.get('lookahead_k')
        except Exception:
            self._known_containers = None
            self._lookahead_k = None
        return None

    def optimize(self, item_list: list):
        """Choose the arrival order. This is the only lever offline mode
        actually gives us -- the return value is a permutation, nothing else
        -- but it is a big one: measured over five scenes, orders drawn at
        random span 11.4 to 52.0 normalized fill on the same scene, and the
        offline scenes average 46.7 against 25.5 for the online ones
        (docs/GAMEPLAN.md #3.1-3.2).

        Search structure: four fixed sort keys, then GRASP-style randomized
        sort keys, then ruin-and-recreate on the incumbent. The last stage is
        the point. A sort key -- any weighting of four item features -- can
        only reach a four-dimensional sliver of the n! orders, and measuring
        the budget showed that sliver is exhausted almost immediately
        (opt=15s, 60s and 150s all score the same). Ruin-and-recreate moves
        in permutation space directly, so extra budget has somewhere to go.
        """
        deadline = time.perf_counter() + OPTIMIZE_TIME_BUDGET
        try:
            specs = [_item_spec(d) for d in item_list]
            all_indices = [s['index'] for s in specs]
            by_index = {s['index']: s for s in specs}
            containers = self._known_containers
            if not containers:
                return all_indices
            any_priority_container = any(c.get('is_prioritized') for c in containers)

            def sort_key_volume(s):
                return (1 if s['is_soft'] else 0, -(s['length'] * s['width'] * s['height']))

            def sort_key_footprint(s):
                dims = sorted((s['length'], s['width'], s['height']), reverse=True)
                return (1 if s['is_soft'] else 0, -(dims[0] * dims[1]), -dims[2])

            def sort_key_height(s):
                return (1 if s['is_soft'] else 0, -s['height'],
                        -(s['length'] * s['width'] * s['height']))

            def sort_key_mass(s):
                return (1 if s['is_soft'] else 0, -s['mass'])

            def make_random_key(rng):
                # A random positive weighting over four normalized item
                # features -- GRASP-style multi-start. Each trial is a fully
                # deterministic greedy rollout, just seeded by a different
                # arrival order, so different trials jam in different places.
                # Weights come from an item-list-derived seed, not system
                # entropy, so a scene always searches the same sequence.
                wv, wf, wh, wm = (rng.random() for _ in range(4))
                total = wv + wf + wh + wm or 1.0
                wv, wf, wh, wm = wv / total, wf / total, wh / total, wm / total

                def key(s):
                    vol = s['length'] * s['width'] * s['height']
                    dims = sorted((s['length'], s['width'], s['height']), reverse=True)
                    foot = dims[0] * dims[1]
                    return (1 if s['is_soft'] else 0,
                            -(wv * vol + wf * foot + wh * s['height'] + wm * s['mass']))
                return key

            def rollout(trial_specs, budget_deadline, resume=None):
                """Greedy rollout of one arrival order, stopped where the real
                episode would stop.

                Two things follow from env.step ending the episode on the
                first placement it cannot make.

                *Objective*: what this order is worth is the length of its
                placeable prefix, not the total number of items a
                skip-and-continue rollout manages to fit. Optimising total
                placed rewards orders that recover after a jam, which the
                real run never gets to do.

                *Cost*: everything after the first failure is simulation the
                real episode will never reach -- and it is nearly the whole
                bill, because best_placement is cheap while there is room and
                expensive once there is not (it exhausts every candidate plus
                both grid fallbacks before reporting failure). Measured on
                two offline scenes: a full 45-item rollout costs 6.6s, of
                which 0.0s falls before the first failure; an 80-item one
                costs 13.6s, of which 0.4s. Stopping at the failure makes a
                rollout 10-30x cheaper, which is the difference between ~10
                rollouts per optimize() call and several hundred -- i.e.
                between the ruin-and-recreate phase below never running and
                it actually searching.

                A lookahead pool of k lets the real run step over k-1 awkward
                items before it is truly stuck, so that many deferrals are
                allowed before stopping.

                *Ties*: prefix length is an integer over a 41! search space,
                so it is mostly plateau -- 820 rollouts on R000 touched only
                twelve distinct values of it. What breaks the tie therefore
                decides most of the search, and it is scored by
                packer.pack_metrics: the composite the leaderboard actually
                pays for, not the volume the old tiebreak preferred. See the
                note above OFFLINE_TIEBREAK in packer.py.

                *Resuming*: ruin-and-recreate only ever moves items around
                inside one order, so a trial and the order it was derived
                from agree on a long prefix -- measured on R000, 64% of the
                items and 60% of the wall time. Simulating that prefix again
                is pure waste, so a rollout hands back a trail (its state
                just before each item) and the next one can be told to pick
                up from position `start` of it. ToP (arXiv 2504.04421) caches
                shared search paths for the same reason and reports decision
                time nearly halving.

                `resume` is (start, trail). Every entry is cloned on the way
                in, so a resumed rollout cannot write through to the trail it
                started from and the trail stays reusable.
                """
                allowed_misses = max(1, int(self._lookahead_k or 1))
                if resume is not None:
                    start, base_trail = resume
                    snap_states, snap_ordered, snap_deferred, snap_plan, misses = \
                        base_trail[start]
                    states = {k: v.clone() for k, v in snap_states.items()}
                    ordered, deferred = list(snap_ordered), list(snap_deferred)
                    plan = dict(snap_plan)
                    trail = list(base_trail[:start])
                else:
                    start = 0
                    states = {c['index']: ContainerState(c) for c in containers}
                    ordered, deferred = [], []
                    plan = {}
                    misses = 0
                    trail = []
                keep_trail = bool(packer.OFFLINE_PREFIX_CACHE)
                stopped_at = None
                for pos_i in range(start, len(trial_specs)):
                    spec = trial_specs[pos_i]
                    if keep_trail:
                        trail.append(({k: v.clone() for k, v in states.items()},
                                      list(ordered), list(deferred), dict(plan), misses))
                    if time.perf_counter() > budget_deadline:
                        stopped_at = pos_i
                        break
                    best = None
                    best_score = float('-inf')
                    for relax in (False, True):
                        # Mirror policy(): it falls back to a relaxed search
                        # (priority/soft stacking allowed) before giving up,
                        # so a rollout that does not would call "failure" at
                        # steps the real run survives -- and since the whole
                        # objective is where the first real failure lands,
                        # that mismatch is not a detail.
                        for cidx, cstate in states.items():
                            result = best_placement(cstate, spec, budget_deadline, relax)
                            if result is None:
                                continue
                            pos, orn, score = result
                            # ...and mirror what choose_action adds on top of
                            # best_placement's score, or the rollout picks a
                            # different container than the run it is
                            # predicting. priority_bonus is the one that
                            # actually bites here: it is the only term that
                            # varies across containers for a fixed item, and
                            # it is worth +300/-600, which dominates
                            # everything else in the score. The rest are
                            # constant per item, so they cannot change this
                            # argmax -- they are here so the two scoring
                            # paths stay visibly identical, since the bug was
                            # exactly that they drifted apart.
                            score += packer.priority_bonus(spec, cstate.geom,
                                                           any_priority_container)
                            score += packer.W_ITEM_VOL * (spec['length'] * spec['width']
                                                          * spec['height'])
                            if packer.W_SOFT_DEFER and spec.get('is_soft'):
                                score -= packer.W_SOFT_DEFER
                            if packer.W_PRIO_DEFER and spec.get('is_prioritized'):
                                score -= packer.W_PRIO_DEFER
                            if score > best_score:
                                best_score = score
                                best = (cidx, pos, orn)
                        if best is not None:
                            break
                    if best is None:
                        deferred.append(spec['index'])
                        misses += 1
                        if misses >= allowed_misses:
                            stopped_at = pos_i + 1
                            break
                        continue
                    cidx, pos, orn = best
                    half = geometry.half_extents(spec['length'], spec['width'], spec['height'], orn)
                    if packer.ROLLOUT_SETTLE:
                        # The evaluator drops the item and it settles flush;
                        # keeping the design height here is what made the
                        # rollout rank orders at Spearman +0.27 within a
                        # scene (tools/order_fidelity.py). Never raise the
                        # item, only let it fall.
                        pos = (pos[0], pos[1],
                               min(pos[2], packer.settled_center_z(states[cidx], pos[0],
                                                                   pos[1], half)))
                    states[cidx].commit(pos, half, spec)
                    ordered.append(spec['index'])
                    plan[spec['index']] = (cidx, pos[0], pos[1], orn)
                tail = ([s['index'] for s in trial_specs[stopped_at:]]
                        if stopped_at is not None else [])
                return ((len(ordered), _tiebreak(states)),
                        ordered + deferred + tail, plan, trail)

            def specs_of(order):
                return [by_index[i] for i in order]

            base_keys = [sort_key_volume, sort_key_footprint, sort_key_height, sort_key_mass]

            best_key = (-1, -1.0)
            best_order = list(all_indices)
            best_plan = {}
            # Where the *walk* currently stands, which is what ruin and
            # recreate perturbs. Held apart from the record: `best_*` only
            # ever improves and is what optimize() returns, while `cur_*` is
            # allowed to step sideways so the search can cross the plateau.
            cur_key, cur_order = best_key, best_order
            # The rollout trail we can resume from, and the *input* order it
            # was produced for. Those differ when a rollout defers an item
            # (only possible when the lookahead pool holds more than one), so
            # the trail is matched against the order it was built from rather
            # than against cur_order.
            cur_trail, cur_trail_order = [], []

            def merit(key):
                return key[0] * packer.RRT_ITEM_WORTH + key[1]

            def resume_for(order):
                """Where `order` stops agreeing with the trail we are holding.

                Returns a `resume` argument for rollout(), or None when there
                is nothing worth reusing.
                """
                if not cur_trail or not packer.OFFLINE_PREFIX_CACHE:
                    return None
                n = 0
                for a, b in zip(order, cur_trail_order):
                    if a != b:
                        break
                    n += 1
                n = min(n, len(cur_trail) - 1)
                return (n, cur_trail) if n > 0 else None

            def offer(key, order, plan, t_order=None, trail=None):
                nonlocal best_key, best_order, best_plan, cur_key, cur_order
                nonlocal cur_trail, cur_trail_order
                improved = key > best_key
                if improved:
                    best_key, best_order, best_plan = key, order, plan
                # Record-to-record travel. Accepting only strict improvements
                # -- what this did until 2026-09-08 -- froze the search: 820
                # rollouts on R000 produced four accepted moves, the last at
                # t=9.7s of a 150s budget, so 140s of ruin and recreate ran
                # against an incumbent it could never leave.
                if merit(key) >= merit(best_key) - packer.RRT_DEV:
                    cur_key, cur_order = key, order
                    if trail:
                        cur_trail, cur_trail_order = trail, t_order
                return improved

            # 1. Fixed heuristics, each on an even share of a modest opening
            #    slice. Bounds the worst case if every rollout is slow, and
            #    measures per-rollout cost to size the phases that follow.
            opening_end = min(deadline, time.perf_counter() + OPTIMIZE_TIME_BUDGET * 0.25)
            n_left = len(base_keys)
            trial_times = []
            for key_fn in base_keys:
                if time.perf_counter() > deadline:
                    break
                t0 = time.perf_counter()
                per_trial = min(deadline, t0 + max(opening_end - t0, 0.0) / max(n_left, 1))
                n_left -= 1
                trial_specs = sorted(specs, key=key_fn)
                key, order, plan, trail = rollout(trial_specs, per_trial,
                                                  resume_for([s['index'] for s in trial_specs]))
                trial_times.append(time.perf_counter() - t0)
                offer(key, order, plan, [s['index'] for s in trial_specs], trail)

            avg_trial = (sum(trial_times) / len(trial_times)) if trial_times else 0.5
            rng = random.Random((sum(s['index'] for s in specs) * 2654435761
                                 + int(packer.OFFLINE_SEED_SALT) * 40503) % (2 ** 32))

            def run_trial(trial_specs, adopt=False):
                """Score one arrival order. `adopt` moves the walk onto it
                whatever it scored, which is what makes a restart a restart
                rather than a single trial the record then discards."""
                nonlocal avg_trial, cur_key, cur_order
                t_order = [s['index'] for s in trial_specs]
                t0 = time.perf_counter()
                key, order, plan, trail = rollout(trial_specs, deadline, resume_for(t_order))
                trial_times.append(time.perf_counter() - t0)
                avg_trial = sum(trial_times) / len(trial_times)
                acc = offer(key, order, plan, t_order, trail)
                if adopt:
                    cur_key, cur_order = key, order
                return acc

            # 2. GRASP diversification: a handful of randomized sort keys, to
            #    give ruin-and-recreate a decent incumbent to work from
            #    rather than whichever fixed key happened to win.
            n_diverse = 0
            diverse_cap = 8 if OFFLINE_SEARCH == 'lns' else (200 if OFFLINE_SEARCH == 'grasp' else 0)
            while n_diverse < diverse_cap and deadline - time.perf_counter() > avg_trial * 1.15:
                run_trial(sorted(specs, key=make_random_key(rng)))
                n_diverse += 1

            # 3. Ruin and recreate on the incumbent order. Two ruin operators:
            #    a random segment, and -- targeted at this problem -- the
            #    items the incumbent could not place at all, which get
            #    reinserted early where there is still room for them.
            n_items = len(specs)
            # Restart schedule. `best_*` is global and monotone, so a restart
            # can only ever add; all it resets is where the walk stands.
            lns_start = time.perf_counter()
            lns_span = max(deadline - lns_start, 0.0)
            n_restarts = max(1, int(packer.OFFLINE_RESTARTS))
            restart_i = 0
            while (OFFLINE_SEARCH == 'lns' and n_items > 2
                   and deadline - time.perf_counter() > avg_trial * 1.15):
                if (restart_i + 1 < n_restarts
                        and time.perf_counter() - lns_start
                        > lns_span * (restart_i + 1) / n_restarts):
                    # Next slice: drop the walk somewhere else in the space.
                    # A randomized sort key, not a random permutation --
                    # arrival orders that ignore item shape entirely are so
                    # far off that the walk spends the whole slice climbing
                    # back to where it started.
                    restart_i += 1
                    run_trial(sorted(specs, key=make_random_key(rng)), adopt=True)
                    continue
                order = list(cur_order)
                pos_of = {idx: i for i, idx in enumerate(order)}
                cut = cur_key[0]
                if cut < n_items and rng.random() < 0.5:
                    # Blocked items first: everything from the first failure
                    # on is what the incumbent could not fit. Pull a few of
                    # them forward into the placeable prefix.
                    victims = order[cut:cut + max(1, n_items // 10)]
                else:
                    k = max(1, min(rng.randint(2, max(2, n_items // 8)), n_items - 1))
                    start = rng.randrange(max(1, n_items - k))
                    victims = order[start:start + k]
                vset = set(victims)
                rest = [i for i in order if i not in vset]
                for idx in victims:
                    # Reinsert near, but not exactly at, where it was: local
                    # moves keep the incumbent's structure, which is what
                    # makes this a neighbourhood search rather than a restart.
                    home = pos_of.get(idx, len(rest))
                    lo = max(0, home - max(3, n_items // 6))
                    hi = min(len(rest), home + max(3, n_items // 6))
                    rest.insert(rng.randint(lo, hi) if hi >= lo else len(rest), idx)
                if len(rest) != n_items:
                    continue
                run_trial(specs_of(rest))

            if set(best_order) != set(all_indices):
                return all_indices
            self._plan = best_plan
            self._plan_rank = {idx: i for i, idx in enumerate(best_order)}
            return best_order
        except Exception:
            traceback.print_exc()
            self._plan = {}
            self._plan_rank = {}
            try:
                return [d['index'] for d in item_list]
            except Exception:
                return []

    def policy(self, observation: dict):
        start = time.perf_counter()
        primary_deadline = start + POLICY_TIME_BUDGET * 0.7
        final_deadline = start + POLICY_TIME_BUDGET
        twins = {}
        try:
            container_list = observation.get('container_list') or []
            pool_list = observation.get('pool_list') or []
            if not container_list or not pool_list:
                return _fallback_action(observation, final_deadline)

            container_states = {c['index']: ContainerState(c) for c in container_list}
            pool = [(i, _item_spec(d, i)) for i, d in enumerate(pool_list)]

            # Physics gate. Built fresh from the observation every call
            # rather than carried on self: policy() has to survive the
            # process restart a timeout causes, and rebuilding is only
            # ~20ms. Raw pool_list dicts are handed to the twin, not
            # _item_spec's trimmed copy, because the friction and
            # soft-contact fields are exactly what makes it accurate.
            if USE_SETTLE_TWIN:
                # Imported here, not at module scope: the flag is off, and a
                # disabled experiment must not be able to break startup.
                import twin
                twins = twin.make(container_list,
                                  {i: cs.geom for i, cs in container_states.items()})
            checks = [0]

            def verifier(cidx, spec):
                t = twins.get(cidx)
                if t is None:
                    return None
                raw = pool_list[spec['pool_index']]

                def verify(pos, orn):
                    if (checks[0] >= MAX_SETTLE_CHECKS
                            or time.perf_counter() > primary_deadline):
                        return True      # out of budget: trust the static rule
                    checks[0] += 1
                    try:
                        return t.settles(raw, pos, orn)
                    except Exception:
                        return True
                return verify

            action = choose_action(container_states, pool, primary_deadline,
                                   verifier=verifier if twins else None)
            if action is None:
                return _fallback_action(observation, final_deadline, twins, pool_list)

            pool_idx, cidx, pos, orn = action
            return {
                'item_idx': int(pool_idx),
                'container_idx': int(cidx),
                'place_pos': np.array(pos, dtype=np.float32),
                'orientation': int(orn),
            }
        except Exception:
            traceback.print_exc()
            return _fallback_action(observation, final_deadline)
        finally:
            for t in twins.values():
                t.close()
