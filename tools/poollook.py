"""Depth-2 pool selection.

The shipped choose_action scores every pool item independently and takes the
argmax, which measures as actively harmful: with a pool of 20 it loses to
having no choice at all on 4 of 16 scenes (docs/SURVEY.md #1). This tries
each of the top-m items tentatively and then asks what the pool looks like
afterwards, so the choice is made on the resulting state rather than on the
immediate score alone.
"""
import time
import packer

M_CANDIDATES = 3     # pool items tried tentatively
R_FOLLOW = 4         # remaining pool items probed after each tentative move
BASE_FRACTION = 0.5  # share of the budget spent on the base pass
W_FOLLOW = 0.35      # weight on the best follow-up score


def _best_over_containers(container_states, spec, deadline, relax, any_prio):
    best = None
    for cidx, cs in container_states.items():
        if time.perf_counter() > deadline:
            break
        r = packer.best_placement(cs, spec, deadline, relax)
        if r is None:
            continue
        pos, orn, score = r
        score += packer.priority_bonus(spec, cs.geom, any_prio)
        if best is None or score > best[0]:
            best = (score, cidx, pos, orn)
    return best


def choose_action_depth2(container_states, pool, time_deadline, relax=False):
    now = time.perf_counter()
    total = max(time_deadline - now, 0.0)
    base_deadline = now + total * BASE_FRACTION
    any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())

    n_calls = max(len(pool) * len(container_states), 1)
    per_call = max((base_deadline - now) / n_calls, 0.02)

    cands = []
    for pool_idx, spec in pool:
        t = time.perf_counter()
        if t > base_deadline:
            break
        b = _best_over_containers(container_states, spec,
                                  min(base_deadline, t + per_call * len(container_states)),
                                  relax, any_prio)
        if b is None:
            continue
        score, cidx, pos, orn = b
        vol = spec['length'] * spec['width'] * spec['height']
        cands.append((score + packer.W_ITEM_VOL * vol, score, pool_idx, cidx, pos, orn, spec))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    if len(cands) == 1 or len(pool) == 1 or time.perf_counter() > time_deadline:
        _t, _s, pi, ci, pos, orn, _sp = cands[0]
        return (pi, ci, pos, orn)

    # Depth 2: commit each of the top-m tentatively and look at what is left.
    top = cands[:M_CANDIDATES]
    slice_ = max((time_deadline - time.perf_counter()) / len(top), 0.05)
    best_key = None
    best = None
    for (tot_score, base_score, pi, ci, pos, orn, spec) in top:
        cs = container_states[ci]
        half = packer.half_extents(spec['length'], spec['width'], spec['height'], orn)
        cs.commit(pos, half, spec)
        sub_deadline = min(time_deadline, time.perf_counter() + slice_)
        try:
            # Largest remaining items first: they are the ones that lose their
            # slot first, so they carry most of the information about whether
            # this move has jammed the container.
            rest = sorted((c for c in cands if c[2] != pi),
                          key=lambda c: -(c[6]['length'] * c[6]['width'] * c[6]['height'])
                          )[:R_FOLLOW]
            n_ok = 0
            best_follow = 0.0
            for (_t2, _s2, _pi2, _ci2, _p2, _o2, spec2) in rest:
                if time.perf_counter() > sub_deadline:
                    break
                b = _best_over_containers(container_states, spec2, sub_deadline,
                                          relax, any_prio)
                if b is not None:
                    n_ok += 1
                    best_follow = max(best_follow, b[0])
        finally:
            cs.boxes.pop()
            cs._refresh()
        key = (n_ok, tot_score + W_FOLLOW * best_follow)
        if best_key is None or key > best_key:
            best_key = key
            best = (pi, ci, pos, orn)
        if time.perf_counter() > time_deadline:
            break
    return best


def make_random_pool_chooser(seed):
    """Pick the pool item at random (placement still greedy).

    Used to bound what pool selection is worth at all: running a scene with
    several seeds and taking the best isolates the one degree of freedom the
    pool gives -- which item to take next -- from everything else.
    """
    import random
    rng = random.Random(seed)

    def choose(container_states, pool, time_deadline, relax=False):
        now = time.perf_counter()
        any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
        n_calls = max(len(pool) * len(container_states), 1)
        per_call = max((time_deadline - now) / n_calls, 0.02)
        feasible = []
        for pool_idx, spec in pool:
            t = time.perf_counter()
            if t > time_deadline:
                break
            b = _best_over_containers(
                container_states, spec,
                min(time_deadline, t + per_call * len(container_states)), relax, any_prio)
            if b is not None:
                feasible.append((pool_idx, b[1], b[2], b[3]))
        if not feasible:
            return None
        return feasible[rng.randrange(len(feasible))]
    return choose


ROLLOUT_M = 3        # pool items evaluated by rollout
ROLLOUT_FRACTION = 0.6


def choose_action_rollout(container_states, pool, time_deadline, relax=False):
    """Pick the pool item by rolling the rest of the pool out to its first
    failure, rather than by an immediate score.

    Depth-2 on next-step feasibility measured as no gain: feasibility stays
    high right up until the container jams, so a one-step probe is flat
    exactly where it needs to discriminate (same reason the survival re-rank
    in docs/GAMEPLAN.md #1.4 failed). A rollout to the first failure is the
    quantity that actually differs between choices, and it is affordable --
    best_placement is nearly free while there is still room (a 45-item
    rollout costs 6.6s, of which 0.0s falls before the first failure).
    """
    now = time.perf_counter()
    total = max(time_deadline - now, 0.0)
    base_deadline = now + total * (1.0 - ROLLOUT_FRACTION)
    any_prio = any(cs.geom['is_prioritized'] for cs in container_states.values())
    n_calls = max(len(pool) * len(container_states), 1)
    per_call = max((base_deadline - now) / n_calls, 0.02)

    cands = []
    for pool_idx, spec in pool:
        t = time.perf_counter()
        if t > base_deadline:
            break
        b = _best_over_containers(container_states, spec,
                                  min(base_deadline, t + per_call * len(container_states)),
                                  relax, any_prio)
        if b is None:
            continue
        vol = spec['length'] * spec['width'] * spec['height']
        cands.append((b[0] + packer.W_ITEM_VOL * vol, b[0], pool_idx, b[1], b[2], b[3], spec))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    if len(cands) == 1 or len(pool) == 1 or time.perf_counter() > time_deadline:
        return (cands[0][2], cands[0][3], cands[0][4], cands[0][5])

    top = cands[:ROLLOUT_M]
    slice_ = max((time_deadline - time.perf_counter()) / len(top), 0.05)
    best_key = None
    best = None
    for (tot, base, pi, ci, pos, orn, spec) in top:
        cs = container_states[ci]
        cs.commit(pos, packer.half_extents(spec['length'], spec['width'], spec['height'], orn), spec)
        committed = [ci]
        sub = min(time_deadline, time.perf_counter() + slice_)
        try:
            rest = sorted((c for c in cands if c[2] != pi),
                          key=lambda c: -(c[6]['length'] * c[6]['width'] * c[6]['height']))
            prefix = 0
            for (_t2, _b2, _pi2, _ci2, _p2, _o2, spec2) in rest:
                if time.perf_counter() > sub:
                    break
                b = _best_over_containers(container_states, spec2, sub, relax, any_prio)
                if b is None:
                    break                      # first failure: stop, as env.step would
                _s3, cidx3, pos3, orn3 = b
                container_states[cidx3].commit(
                    pos3, packer.half_extents(spec2['length'], spec2['width'],
                                              spec2['height'], orn3), spec2)
                committed.append(cidx3)
                prefix += 1
        finally:
            for cidx in reversed(committed):
                container_states[cidx].boxes.pop()
                container_states[cidx]._refresh()
        key = (prefix, tot)
        if best_key is None or key > best_key:
            best_key = key
            best = (pi, ci, pos, orn)
        if time.perf_counter() > time_deadline:
            break
    return best
