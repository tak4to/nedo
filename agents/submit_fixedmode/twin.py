"""A private replica of the evaluator's third gate.

geometry.py mirrors the first two gates -- inclusion and the transport
sweep -- exactly, because both are pure geometry. The third one is not:
``validator.place_item`` warps the box to its target, runs
``settle_wait_step`` (300) physics steps, and ends the episode if the box
has moved more than 0.3 m or tipped more than 45 degrees. packer's
``_footprint_supported`` is a static-stability stand-in for that, and it
is wrong in both directions -- too strict and the search reports "nothing
fits" while the container is half empty, too loose and boxes topple.

Every input the real settle step consumes is in the observation: the
container's plane equations, and each packed item's pose, mass, friction,
restitution and (for soft items) contact softening. So instead of
approximating the gate we can *run* it, in a private DIRECT-mode PyBullet
world of our own.

Calibration against 12 episodes that really died on this gate: with the
soft-item contact parameters applied it flags 12 of 12 at the evaluator's
own 300 steps. At 100 steps -- 71 ms including rebuilding the world --
the collapse is already well separated from the 1-2 cm of ordinary
settling drift, so a 0.05 m / 8 degree threshold catches 6 of 6 while
falsely rejecting 9 of 196 placements the evaluator went on to accept.
That asymmetry is the whole point: a false reject costs us the
second-best placement, a missed topple costs the entire episode.

Everything here is best-effort. If PyBullet cannot be imported or a
world cannot be built, `make` returns None and the caller silently falls
back to the static criterion alone.
"""
import math

ORNS = [
    (0.0, 0.0, 0.0),
    (math.pi / 2, 0.0, 0.0),
    (0.0, math.pi / 2, 0.0),
    (0.0, 0.0, math.pi / 2),
    (0.0, math.pi / 2, math.pi / 2),
    (math.pi / 2, 0.0, math.pi / 2),
]

SETTLE_STEPS = 100
DISP_LIMIT = 0.05
ANGLE_LIMIT_DEG = 8.0


def _quat_from_z(n):
    """Quaternion rotating +Z onto the unit vector n."""
    nx, ny, nz = n
    if nz > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if nz < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sqrt((1.0 + nz) * 2.0)
    return (-ny / s, nx / s, 0.0, s / 2.0)


def _dyn(it):
    """The dynamics Item.create applies, including the soft-item contact
    softening. Leaving that out simulates a squishy bag as a rigid one,
    which hid 7 of the 12 real collapses during calibration."""
    d = {}
    for key, default in (('lateralFriction', 0.5), ('rollingFriction', 0.01),
                         ('spinningFriction', 0.01), ('restitution', 0.1),
                         ('angularDamping', None)):
        v = it.get(key, default)
        if v is not None:
            d[key] = v
    if it.get('is_soft'):
        for key in ('contactStiffness', 'contactDamping', 'linearDamping'):
            if it.get(key) is not None:
                d[key] = it[key]
    return d


class Twin:
    def __init__(self, p, cdict, static_boxes):
        self.p = p
        import pybullet_utils.bullet_client as bc
        self.cl = bc.BulletClient(connection_mode=p.DIRECT)
        self.cl.setGravity(0, 0, -9.8)          # env.py uses -9.8, not -9.81
        self.cl.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.ox = cdict['center'][0]
        big, thin = 4.0, 0.05
        for n, pt in zip(cdict['n_vecs'], cdict['points']):
            n = [float(v) for v in n]
            # Outward -Y is the doorway. The real container is an open cup,
            # so a box that ends up leaning on the opening falls out of it.
            # Walling it off would turn exactly the collapses we are looking
            # for into apparent successes.
            if n[1] < -0.99:
                continue
            pt = [float(pt[0]) - self.ox, float(pt[1]), float(pt[2])]
            col = self.cl.createCollisionShape(p.GEOM_BOX, halfExtents=[big, big, thin])
            self.cl.createMultiBody(
                0, col,
                basePosition=[pt[i] + n[i] * thin for i in range(3)],
                baseOrientation=_quat_from_z(n))
        for c, h in static_boxes:
            col = self.cl.createCollisionShape(p.GEOM_BOX, halfExtents=list(h))
            self.cl.createMultiBody(0, col, basePosition=list(c))
        for it in cdict.get('packed_items', []):
            if it.get('pos') is None or it.get('orn') is None:
                continue
            col = self.cl.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[it['length'] / 2.0, it['width'] / 2.0, it['height'] / 2.0])
            bid = self.cl.createMultiBody(
                it['mass'], col,
                basePosition=(it['pos'][0] - self.ox, it['pos'][1], it['pos'][2]),
                baseOrientation=it['orn'])
            self.cl.changeDynamics(bid, -1, **_dyn(it))

    def settles(self, spec, pos_local, orn_idx, steps=SETTLE_STEPS):
        """True if this placement survives the settle step."""
        p = self.p
        quat = p.getQuaternionFromEuler(ORNS[orn_idx])
        col = self.cl.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[spec['length'] / 2.0, spec['width'] / 2.0, spec['height'] / 2.0])
        bid = self.cl.createMultiBody(spec.get('mass', 1.0), col,
                                      basePosition=pos_local, baseOrientation=quat)
        self.cl.changeDynamics(bid, -1, **_dyn(spec))
        # Snapshot *after* adding the candidate: restoring a state that
        # predates a body leaves the importer with fewer bodies than the
        # world has and the restore silently fails. Stepping moves the
        # already-packed boxes too, so this has to be undone either way.
        state = self.cl.saveState()
        try:
            for _ in range(steps):
                self.cl.stepSimulation()
            fp, fo = self.cl.getBasePositionAndOrientation(bid)
            disp = math.dist(fp, pos_local)
            dot = min(1.0, abs(sum(a * b for a, b in zip(quat, fo))))
            angle = math.degrees(2 * math.acos(dot))
            return disp <= DISP_LIMIT and angle <= ANGLE_LIMIT_DEG
        finally:
            self.cl.restoreState(state)
            self.cl.removeState(state)
            self.cl.removeBody(bid)

    def close(self):
        try:
            self.cl.disconnect()
        except Exception:
            pass


def make(container_dicts, geoms):
    """Best-effort twin per container index, or {} if physics is unavailable."""
    try:
        import pybullet as p
    except Exception:
        return {}
    out = {}
    for cd in container_dicts:
        try:
            out[cd['index']] = Twin(p, cd, geoms[cd['index']]['static_boxes'])
        except Exception:
            for t in out.values():
                t.close()
            return {}
    return out
