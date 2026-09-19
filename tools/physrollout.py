"""A rollout that settles each placement in physics before deciding the next.

optimize()'s own rollout commits every item at a computed height and moves
on. That is what makes it cheap, and it is why it ranks two arrival orders
for the same scene at only Spearman +0.49 (tools/order_fidelity.py) -- and
why every attempt to buy more of it (record-to-record travel, a prefix
cache, multi-start) lost to the winner's curse.

Two-stage selection is the way out, and its ceiling is measured: letting an
oracle pick from the cheap rollout's top three is worth +2.12 composite. A
judge that only *settles the finished packing* was tried and came out worse
than no judge at all (rho 0.234 against 0.337, tools/stage2_probe.py) -- the
diagnosis being that dropping a whole planned stack in at once is different
physics from placing one item at a time.

So this places one item at a time. Each step re-derives the best placement
against the state physics actually left behind, exactly as policy() re-derives
from the observation, then settles the new item and reads its real pose back.
It also scores stability, which the cheap rollout is structurally blind to.

Cost is roughly 0.3s per item, so one order is 10-30s: affordable for a
handful of finalists, never for the search itself.
"""
import math
import os
import statistics
import sys

METRICS = ['fill', 'cog_score', 'stability_score', 'placement_score', 'soft_item_score']


def _half_from_quat(cl, spec, quat):
    R = cl.getMatrixFromQuaternion(quat)
    l, w, h = spec['length'], spec['width'], spec['height']
    return (
        (abs(R[0]) * l + abs(R[1]) * w + abs(R[2]) * h) / 2.0,
        (abs(R[3]) * l + abs(R[4]) * w + abs(R[5]) * h) / 2.0,
        (abs(R[6]) * l + abs(R[7]) * w + abs(R[8]) * h) / 2.0,
    )


def run(container_dicts, geoms, specs_in_order, packer, twin_mod, geometry,
        allowed_misses=1, settle_steps=80, disp_threshold=0.3, shake=True,
        deadline=None):
    """Physics-in-the-loop rollout. Returns (n_placed, composite, detail)."""
    import pybullet as p
    import time as _t

    twins = {}
    states = {}
    for cd in container_dicts:
        ci = cd['index']
        twins[ci] = twin_mod.Twin(p, cd, geoms[ci]['static_boxes'])
        states[ci] = packer.ContainerState(cd)
    any_prio_c = any(g['is_prioritized'] for g in geoms.values())
    placed = []          # (cidx, spec, settled pos, half)
    bodies = []          # (cidx, body id, spec)
    n_placed = 0
    misses = 0
    try:
        for spec in specs_in_order:
            if deadline is not None and _t.perf_counter() > deadline:
                break
            best, best_score = None, float('-inf')
            for relax in (False, True):
                for ci, cs in states.items():
                    r = packer.best_placement(cs, spec, deadline, relax)
                    if r is None:
                        continue
                    pos, orn, score = r
                    # mirror choose_action, as agent.rollout does
                    score += packer.priority_bonus(spec, cs.geom, any_prio_c)
                    score += packer.W_ITEM_VOL * (spec['length'] * spec['width']
                                                  * spec['height'])
                    if packer.W_SOFT_DEFER and spec.get('is_soft'):
                        score -= packer.W_SOFT_DEFER
                    if packer.W_PRIO_DEFER and spec.get('is_prioritized'):
                        score -= packer.W_PRIO_DEFER
                    if score > best_score:
                        best_score, best = score, (ci, pos, orn)
                if best is not None:
                    break
            if best is None:
                misses += 1
                if misses >= allowed_misses:
                    break
                continue
            ci, pos, orn = best
            bid, fp, fo, disp = twins[ci].place_and_settle(spec, pos, orn, settle_steps)
            if disp > disp_threshold:
                # validator.place_item removes it and ends the episode
                break
            half = _half_from_quat(twins[ci].cl, spec, fo)
            states[ci].commit(fp, half, spec)
            placed.append((ci, spec, fp, half))
            bodies.append((ci, bid, spec))
            n_placed += 1

        # --- score the five metrics off the settled state -------------------
        disp_all = []
        if shake and bodies:
            before = {}
            for ci, bid, _ in bodies:
                before[(ci, bid)] = twins[ci].cl.getBasePositionAndOrientation(bid)[0]
            a = 9.8 * 0.30
            for _ in range(2):
                for gx, gy in ((a, 0), (-a, 0), (0, a), (0, -a)):
                    for t in twins.values():
                        t.cl.setGravity(gx, gy, -9.8)
                    for _ in range(150):
                        for t in twins.values():
                            t.cl.stepSimulation()
            for t in twins.values():
                t.cl.setGravity(0, 0, -9.8)
            for _ in range(200):
                for t in twins.values():
                    t.cl.stepSimulation()
            for ci, bid, _ in bodies:
                fp = twins[ci].cl.getBasePositionAndOrientation(bid)[0]
                disp_all.append(math.dist(fp, before[(ci, bid)]))
    finally:
        for t in twins.values():
            t.close()

    if not placed:
        return 0, 0.0, {}
    denom = sum(packer.usable_volume(geoms[cd['index']]) for cd in container_dicts)
    height = max(g['height'] for g in geoms.values())
    counted = mass = moment = 0.0
    boxes = []
    for ci, spec, pos, half in placed:
        v = 8.0 * half[0] * half[1] * half[2]
        if abs((pos[2] - half[2]) - geoms[ci]['floor_struct_z']) >= 0.03:
            counted += v
        m = spec.get('mass', 1.0)
        mass += m
        moment += m * pos[2]
        boxes.append((ci, {'center': pos, 'half': half, 'mass': m,
                           'is_soft': spec.get('is_soft', False),
                           'is_prioritized': spec.get('is_prioritized', False)}))

    def cls(attr, check):
        members = [(c, b) for c, b in boxes if b[attr]]
        if not members:
            return 100.0
        bad = 0
        for c, b in members:
            top = b['center'][2] + b['half'][2]
            for oc, o in boxes:
                if o is b or oc != c or o[attr] == b[attr]:
                    continue
                if o['center'][2] - o['half'][2] < top - 0.02:
                    continue
                if (abs(o['center'][0] - b['center'][0]) < o['half'][0] + b['half'][0] and
                        abs(o['center'][1] - b['center'][1]) < o['half'][1] + b['half'][1]):
                    bad += 1
                    break
        wrong = sum(1 for c, b in members if not geoms[c]['is_prioritized']) if check else 0
        return 100.0 * (1.0 - (bad + wrong) / (2.0 * len(members)))

    mean_d = statistics.mean(disp_all) if disp_all else 0.0
    detail = {
        'fill': min(100.0 * counted / denom, 100.0) if denom else 0.0,
        'cog_score': max(0.0, min(100.0, 100.0 * (1.0 - (moment / mass) / height)))
                     if mass and height else 0.0,
        'stability_score': 100.0 * math.exp(-mean_d / 0.10) if shake else 0.0,
        'placement_score': cls('is_prioritized', any_prio_c),
        'soft_item_score': cls('is_soft', False),
    }
    keys = METRICS if shake else [k for k in METRICS if k != 'stability_score']
    return n_placed, sum(detail[k] for k in keys) / len(keys), detail
