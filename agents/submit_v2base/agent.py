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
from packer import ContainerState, best_placement, choose_action  # noqa: E402

POLICY_TIME_BUDGET = 5.0
OPTIMIZE_TIME_BUDGET = 150.0
INIT_TIME_BUDGET = 8.0


def _item_spec(d):
    return {
        'index': d.get('index'),
        'length': float(d['length']), 'width': float(d['width']), 'height': float(d['height']),
        'mass': float(d.get('mass', 1.0)),
        'is_prioritized': bool(d.get('is_prioritized', False)),
        'is_soft': bool(d.get('is_soft', False)),
    }


def _fallback_action(observation, deadline=None):
    """Last-resort action when the primary search finds nothing: relax the
    priority/soft stacking constraint (a scoring penalty is far cheaper
    than ending the episode) and search again. Always oracle-checked
    before being returned, and never allowed to raise."""
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
        pool = [(i, _item_spec(d)) for i, d in enumerate(pool_list)]

        action = choose_action(container_states, pool, deadline, relax=True)
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

    def get_init_states(self, init_states: dict):
        try:
            self._known_containers = init_states.get('container_list', [])
            self._lookahead_k = init_states.get('lookahead_k')
        except Exception:
            self._known_containers = None
            self._lookahead_k = None
        return None

    def optimize(self, item_list: list):
        deadline = time.perf_counter() + OPTIMIZE_TIME_BUDGET
        try:
            specs = [_item_spec(d) for d in item_list]
            all_indices = [s['index'] for s in specs]
            containers = self._known_containers
            if not containers:
                return all_indices

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
                # features. This is a GRASP-style multi-start: each trial
                # is a fully deterministic greedy simulation (same
                # best_placement used everywhere else), just seeded by a
                # different, randomly-weighted arrival order, so different
                # trials tend to jam in different places. We keep the best
                # one. Weights come from a fixed, item-list-derived seed
                # (not system entropy), so a given scene always searches
                # the same sequence of trials -- reproducible, not noisy.
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

            def simulate(key_fn, budget_deadline):
                trial_specs = sorted(specs, key=key_fn)
                states = {c['index']: ContainerState(c) for c in containers}
                ordered, deferred = [], []
                placed_volume = 0.0
                for spec in trial_specs:
                    if time.perf_counter() > budget_deadline:
                        deferred.append(spec['index'])
                        continue
                    best = None
                    best_score = float('-inf')
                    for cidx, cstate in states.items():
                        result = best_placement(cstate, spec, budget_deadline)
                        if result is None:
                            continue
                        pos, orn, score = result
                        if score > best_score:
                            best_score = score
                            best = (cidx, pos, orn)
                    if best is None:
                        deferred.append(spec['index'])
                        continue
                    cidx, pos, orn = best
                    hx_hy_hz = geometry.half_extents(spec['length'], spec['width'], spec['height'], orn)
                    states[cidx].commit(pos, hx_hy_hz, spec)
                    ordered.append(spec['index'])
                    placed_volume += spec['length'] * spec['width'] * spec['height']
                return ordered, deferred, placed_volume

            base_keys = [sort_key_volume, sort_key_footprint, sort_key_height, sort_key_mass]

            best_result = list(all_indices)
            best_placed = -1
            best_volume = -1.0

            # Fixed heuristics first, each on an even share of a modest
            # opening slice of the budget -- this bounds worst case if
            # every trial is slow, and gives a real measurement of
            # per-trial cost to size the randomized phase that follows.
            opening_budget_end = min(deadline, time.perf_counter() + OPTIMIZE_TIME_BUDGET * 0.25)
            n_left = len(base_keys)
            trial_times = []
            for key_fn in base_keys:
                if time.perf_counter() > deadline:
                    break
                t0 = time.perf_counter()
                per_trial_deadline = min(deadline, t0 + max(opening_budget_end - t0, 0.0) / max(n_left, 1))
                n_left -= 1
                ordered, deferred, vol = simulate(key_fn, per_trial_deadline)
                trial_times.append(time.perf_counter() - t0)
                if (len(ordered) > best_placed or
                        (len(ordered) == best_placed and vol > best_volume)):
                    best_placed, best_volume = len(ordered), vol
                    best_result = ordered + deferred

            # Randomized phase: spend whatever budget is left on as many
            # extra GRASP-style trials as the measured per-trial cost
            # says will fit, each simulated in full (not truncated), with
            # a safety margin so the last trial can't overrun the deadline.
            avg_trial_time = (sum(trial_times) / len(trial_times)) if trial_times else 0.5
            rng = random.Random(sum(s['index'] for s in specs) * 2654435761 % (2**32))
            n_random = 0
            while True:
                remaining = deadline - time.perf_counter()
                if remaining < avg_trial_time * 1.15 or n_random >= 200:
                    break
                trial_deadline = time.perf_counter() + remaining  # full remaining slice
                t0 = time.perf_counter()
                ordered, deferred, vol = simulate(make_random_key(rng), trial_deadline)
                trial_times.append(time.perf_counter() - t0)
                avg_trial_time = sum(trial_times) / len(trial_times)
                n_random += 1
                if (len(ordered) > best_placed or
                        (len(ordered) == best_placed and vol > best_volume)):
                    best_placed, best_volume = len(ordered), vol
                    best_result = ordered + deferred

            if set(best_result) != set(all_indices):
                return all_indices
            return best_result
        except Exception:
            try:
                return [d['index'] for d in item_list]
            except Exception:
                return []

    def policy(self, observation: dict):
        start = time.perf_counter()
        primary_deadline = start + POLICY_TIME_BUDGET * 0.7
        final_deadline = start + POLICY_TIME_BUDGET
        try:
            container_list = observation.get('container_list') or []
            pool_list = observation.get('pool_list') or []
            if not container_list or not pool_list:
                return _fallback_action(observation, final_deadline)

            container_states = {c['index']: ContainerState(c) for c in container_list}
            pool = [(i, _item_spec(d)) for i, d in enumerate(pool_list)]

            action = choose_action(container_states, pool, primary_deadline)
            if action is None:
                return _fallback_action(observation, final_deadline)

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
