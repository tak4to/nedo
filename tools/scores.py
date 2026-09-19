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


PHYSICS_DT = 1.0 / 240.0  # pybullet default; env.py never calls setTimeStep

# Normalisation constants for the force/energy components below. Following
# the same convention as the pre-existing disp_score (mean_d/0.10, i.e. the
# constant is picked so the *current agent's typical value* lands at
# exp(-1)=37: enough headroom to see a change move the score in either
# direction). Measured on 10 scenes from scenes_off.json with the current
# agent (tools/calib_stability.py): shake_mean_peak_force ranged 658-1177N
# (median 960), shake_mean_energy ranged 2.33-3.73J (median 2.71) --
# both far tighter than displacement's spread (0.099-0.171m), which on its
# own is a first data point (§P1/P2): this agent's episodes are more
# uniform in force/energy than in how far things end up drifting.
#
# NOT calibrated against the real evaluator -- its formula and weights are
# private (docs/comp_determin.md, docs/2026-09-13-survey戦略.md P0/P1).
# Revisit the moment a real submission's stability_score is known, and
# re-run tools/calib_stability.py if the agent changes enough that the
# median force/energy drifts.
FORCE_SCALE = 960.0     # N; mean per-item peak contact force during shake
ENERGY_SCALE = 2.71     # J; mean per-item integrated kinetic energy during shake


def stability_score(env, tilt=0.30, cycles=2, steps_per=150, settle=200, sample_every=4):
    """README: 'how little the load moves / collapses when the container is
    shaken'. Container bodies are static, so the shake is applied as a
    rotating lateral acceleration (equivalent to tilting the whole rig), with
    the door lid fitted first via Container.create_cap -- which the local
    runner never calls, and which exists precisely for this test.

    docs/comp_determin.md (the official evaluation page, transcribed
    verbatim) names three observables for this test, not one: how far each
    item drifted once the shaking stops, the force on items *during* the
    shake, and the kinetic energy generated during the shake. Until
    2026-09-13 this only computed the first (`shake_mean_disp` /
    `shake_n_moved`); the composite `stability_score` it returned was
    therefore built from 1 of 3 official signals. This adds the other two:

      * force: at each sampled step, the sum of contact-normal forces on
        each item, tracked at its peak over the whole shake.
      * energy: at each sampled step, 0.5*m*|v|^2 + 0.5*I_iso*|w|^2 per item
        (I_iso = mean of the local inertia diagonal -- an isotropic stand-in
        for the true tensor, since we don't track orientation-dependent
        inertia here), trapezoidally integrated over the shake.

    Neither the real combination formula nor the component weights are
    published, so `stability_score` here is an equal-weight mean of three
    independent exp-decay proxies (one per observable) -- a *placeholder*
    weighting, not a fit to anything. Returns a dict; see FORCE_SCALE /
    ENERGY_SCALE above for the two constants this introduces and how they
    were picked.
    """
    bs = _boxes(env)
    if not bs:
        return dict(stability_score=0.0, shake_mean_disp=0.0, shake_n_moved=0,
                     shake_mean_peak_force=0.0, shake_mean_energy=0.0,
                     disp_score=0.0, force_score=0.0, energy_score=0.0,
                     per_item=[])
    pid_of = {b['it'].pybullet_id: b for b in bs}
    before = {pid: b['pos'].copy() for pid, b in pid_of.items()}
    peak_force = {pid: 0.0 for pid in pid_of}
    energy_integral = {pid: 0.0 for pid in pid_of}
    prev_ke = {pid: 0.0 for pid in pid_of}
    inertia_iso = {}
    for pid in pid_of:
        try:
            di = env.client.getDynamicsInfo(pid, -1)
            inertia_iso[pid] = float(sum(di[2]) / 3.0)
        except Exception:
            inertia_iso[pid] = 0.0

    def sample(dt_since_last):
        # Contact forces: one global query per sample, bucketed by body id.
        # Static bodies (container/cap) have negative or fixed ids and are
        # not in pid_of, so this only accumulates item-on-item and
        # item-on-container forces attributed to the *item* side.
        for cp in env.client.getContactPoints():
            fn = cp[9]  # normal force
            a_id, b_id = cp[1], cp[2]
            if a_id in peak_force:
                peak_force[a_id] = max(peak_force[a_id], fn)
            if b_id in peak_force:
                peak_force[b_id] = max(peak_force[b_id], fn)
        for pid in pid_of:
            lin, ang = env.client.getBaseVelocity(pid)
            m = pid_of[pid]['it'].mass
            v2 = sum(x * x for x in lin)
            w2 = sum(x * x for x in ang)
            ke = 0.5 * m * v2 + 0.5 * inertia_iso[pid] * w2
            # trapezoidal integral in energy*time (J*s), then /steps_per_cycle
            # below to report a per-cycle-comparable "mean energy" scale.
            energy_integral[pid] += 0.5 * (ke + prev_ke[pid]) * dt_since_last
            prev_ke[pid] = ke

    caps = [c.create_cap(env.client) for c in env.container_manager.containers]
    g = 9.8
    a = g * tilt
    try:
        step_i = 0
        for _ in range(cycles):
            for (gx, gy) in ((a, 0), (-a, 0), (0, a), (0, -a)):
                env.client.setGravity(gx, gy, -g)
                for _ in range(steps_per):
                    env.client.stepSimulation()
                    step_i += 1
                    if step_i % sample_every == 0:
                        sample(sample_every * PHYSICS_DT)
        env.client.setGravity(0, 0, -g)
        for _ in range(settle):
            env.client.stepSimulation()
        disp = []
        per_item = []
        for pid, b in pid_of.items():
            pos, _ = b['it'].get_pose(env.client)
            if pos is None:
                continue
            d = float(np.linalg.norm(np.array(pos) - before[pid]))
            disp.append(d)
            per_item.append(dict(index=b['it'].index, disp=d,
                                 peak_force=peak_force[pid],
                                 energy=energy_integral[pid],
                                 mass=b['it'].mass, is_soft=b['it'].is_soft,
                                 is_prioritized=b['it'].is_prioritized))
    finally:
        for cid in caps:
            try:
                env.client.removeBody(cid)
            except Exception:
                pass
        env.client.setGravity(0, 0, -g)
    if not disp:
        return dict(stability_score=0.0, shake_mean_disp=0.0, shake_n_moved=0,
                     shake_mean_peak_force=0.0, shake_mean_energy=0.0,
                     disp_score=0.0, force_score=0.0, energy_score=0.0,
                     per_item=[])
    mean_d = sum(disp) / len(disp)
    n_moved = sum(1 for d in disp if d > 0.05)
    mean_force = sum(peak_force.values()) / len(peak_force)
    mean_energy = sum(energy_integral.values()) / len(energy_integral)
    # 0 displacement -> 100, 10cm mean -> ~37, 20cm -> ~14
    disp_score = 100.0 * math.exp(-mean_d / 0.10)
    force_score = 100.0 * math.exp(-mean_force / FORCE_SCALE)
    energy_score = 100.0 * math.exp(-mean_energy / ENERGY_SCALE)
    composite = (disp_score + force_score + energy_score) / 3.0
    return dict(stability_score=composite, shake_mean_disp=mean_d, shake_n_moved=n_moved,
                 shake_mean_peak_force=mean_force, shake_mean_energy=mean_energy,
                 disp_score=disp_score, force_score=force_score, energy_score=energy_score,
                 per_item=per_item)


def all_scores(env, per_item=False):
    cog, cog_z = cog_score(env)
    pl = placement_scores(env)
    stab = stability_score(env)
    out = dict(cog_score=cog, cog_z=cog_z,
               placement_score=pl['prio'][0], soft_item_score=pl['soft'][0],
               n_prio_bad=pl['prio'][1], n_prio_wrong=pl['prio'][2], n_soft_bad=pl['soft'][1],
               stability_score=stab['stability_score'],
               shake_mean_disp=stab['shake_mean_disp'], shake_n_moved=stab['shake_n_moved'],
               shake_mean_peak_force=stab['shake_mean_peak_force'],
               shake_mean_energy=stab['shake_mean_energy'],
               stability_disp_score=stab['disp_score'],
               stability_force_score=stab['force_score'],
               stability_energy_score=stab['energy_score'])
    if per_item:
        out['stability_per_item'] = stab['per_item']
    return out
