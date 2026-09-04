"""Local proxies for the four scores the distributed evaluator does not
implement (cog / stability / placement / soft).

None of these are the official formulas -- those are not published -- but
each is monotone in the same direction as the description in README.md, so
they can rank two agents even if the absolute numbers differ.
"""
import math
import numpy as np


def _boxes(env):
    out = []
    for c in env.container_manager.containers:
        for it in c.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            R = np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            h = np.abs(R) @ np.array([it.length / 2, it.width / 2, it.height / 2])
            out.append(dict(it=it, c=c, pos=np.array(pos), h=h))
    return out


def cog_score(env):
    """README: 'how low the centre of gravity of the whole load sits'.
    Normalised against the container's usable height."""
    bs = _boxes(env)
    if not bs:
        return 0.0, 0.0
    m = sum(b['it'].mass for b in bs)
    z = sum(b['it'].mass * b['pos'][2] for b in bs) / m
    H = max(c.height for c in env.container_manager.containers)
    return max(0.0, min(100.0, 100.0 * (1.0 - z / H))), float(z)


def placement_scores(env):
    """README: a priority item (resp. soft item) is penalised when an item of
    a *different* class rests on top of it; same-class stacking is free. A
    priority item in the wrong container is penalised too."""
    bs = _boxes(env)
    any_prio_c = any(c.is_prioritized for c in env.container_manager.containers)
    res = {}
    for kind in ('prio', 'soft'):
        members = [b for b in bs if (b['it'].is_prioritized if kind == 'prio' else b['it'].is_soft)]
        if not members:
            res[kind] = (100.0, 0, 0)
            continue
        bad = 0
        for b in members:
            top = b['pos'][2] + b['h'][2]
            covered = False
            for o in bs:
                if o is b or o['c'] is not b['c']:
                    continue
                same = ((o['it'].is_prioritized == b['it'].is_prioritized) if kind == 'prio'
                        else (o['it'].is_soft == b['it'].is_soft))
                if same:
                    continue
                if o['pos'][2] - o['h'][2] < top - 0.02:
                    continue
                if (abs(o['pos'][0] - b['pos'][0]) < o['h'][0] + b['h'][0] and
                        abs(o['pos'][1] - b['pos'][1]) < o['h'][1] + b['h'][1]):
                    covered = True
                    break
            if covered:
                bad += 1
        wrong = 0
        if kind == 'prio' and any_prio_c:
            wrong = sum(1 for b in members if not b['c'].is_prioritized)
        res[kind] = (100.0 * (1.0 - (bad + wrong) / (2 * len(members))), bad, wrong)
    return res


def stability_score(env, tilt=0.30, cycles=2, steps_per=150, settle=200):
    """README: 'how little the load moves / collapses when the container is
    shaken'. Container bodies are static, so the shake is applied as a
    rotating lateral acceleration (equivalent to tilting the whole rig), with
    the door lid fitted first via Container.create_cap -- which the local
    runner never calls, and which exists precisely for this test."""
    bs = _boxes(env)
    if not bs:
        return 0.0, 0.0, 0
    before = {b['it'].pybullet_id: b['pos'].copy() for b in bs}
    caps = [c.create_cap(env.client) for c in env.container_manager.containers]
    g = 9.8
    a = g * tilt
    try:
        for _ in range(cycles):
            for (gx, gy) in ((a, 0), (-a, 0), (0, a), (0, -a)):
                env.client.setGravity(gx, gy, -g)
                for _ in range(steps_per):
                    env.client.stepSimulation()
        env.client.setGravity(0, 0, -g)
        for _ in range(settle):
            env.client.stepSimulation()
        disp = []
        for b in bs:
            pid = b['it'].pybullet_id
            if pid is None:
                continue
            pos, _ = b['it'].get_pose(env.client)
            if pos is None:
                continue
            disp.append(float(np.linalg.norm(np.array(pos) - before[pid])))
    finally:
        for cid in caps:
            try:
                env.client.removeBody(cid)
            except Exception:
                pass
        env.client.setGravity(0, 0, -g)
    if not disp:
        return 0.0, 0.0, 0
    mean_d = sum(disp) / len(disp)
    n_moved = sum(1 for d in disp if d > 0.05)
    # 0 displacement -> 100, 10cm mean -> ~37, 20cm -> ~14
    score = 100.0 * math.exp(-mean_d / 0.10)
    return score, mean_d, n_moved


def all_scores(env):
    cog, cog_z = cog_score(env)
    pl = placement_scores(env)
    stab, mean_d, n_moved = stability_score(env)
    return dict(cog_score=cog, cog_z=cog_z,
                placement_score=pl['prio'][0], soft_item_score=pl['soft'][0],
                n_prio_bad=pl['prio'][1], n_prio_wrong=pl['prio'][2], n_soft_bad=pl['soft'][1],
                stability_score=stab, shake_mean_disp=mean_d, shake_n_moved=n_moved)
