"""How expensive is a private PyBullet 'digital twin' of the settle gate?

validator.place_item warps the box straight to target_pos (no drop), runs
settle_wait_step=300 stepSimulation calls, and fails the episode if that
box moved >0.3m or tipped >45deg. Everything needed to replicate it is in
the observation: container plane equations + every packed item's pose,
mass and friction coefficients.
"""
import os, sys, json, time, math
ROOT = '/home/takato/comp/nedo'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'agents', 'submit')); sys.path.insert(0, '.')
import pybullet as p
import pybullet_utils.bullet_client as bc


def _dyn(it):
    """Exactly the dynamics Item.create applies -- including the soft-item
    contact softening, without which a squishy bag is simulated as a rigid
    one and every collapse it causes is invisible to the twin."""
    d = {}
    for k, dv in (('lateralFriction', 0.5), ('rollingFriction', 0.01),
                  ('spinningFriction', 0.01), ('restitution', 0.1),
                  ('angularDamping', None)):
        v = it.get(k, dv)
        if v is not None:
            d[k] = v
    if it.get('is_soft'):
        for k in ('contactStiffness', 'contactDamping', 'linearDamping'):
            if it.get(k) is not None:
                d[k] = it[k]
    return d


def _quat_from_z(n):
    """Quaternion rotating +Z onto the unit vector n."""
    nx, ny, nz = n
    if nz > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if nz < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = -ny, nx, 0.0          # cross((0,0,1), n)
    s = math.sqrt((1.0 + nz) * 2.0)
    return (ax / s, ay / s, az / s, s / 2.0)


class Twin:
    """Private DIRECT-mode replica of the settle gate.

    The container is rebuilt from the observation's own plane equations
    (points + outward normals), one thin slab per plane, each *oriented*
    by its normal -- the chamfer plane's normal is diagonal, so an
    axis-aligned slab is not an approximation of it, it is a different
    wall in a different place.

    The door plane is deliberately skipped: the real container is an open
    cup, so a box that ends up leaning on the doorway falls out instead of
    being held. Walling it off turns exactly the collapses we are trying
    to detect into apparent successes. The static shelf plates are real
    bodies in the evaluator and are added too.
    """
    def __init__(self, cdict, geom=None):
        self.cl = bc.BulletClient(connection_mode=p.DIRECT)
        self.cl.setGravity(0, 0, -9.8)          # env.py uses -9.8, not -9.81
        self.cl.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.ox = cdict['center'][0]
        big, thin = 4.0, 0.05
        for n, pt in zip(cdict['n_vecs'], cdict['points']):
            n = [float(v) for v in n]
            if n[1] < -0.99:          # outward -Y == the open doorway
                continue
            pt = [float(pt[0]) - self.ox, float(pt[1]), float(pt[2])]
            centre = [pt[i] + n[i] * thin for i in range(3)]
            col = self.cl.createCollisionShape(p.GEOM_BOX, halfExtents=[big, big, thin])
            self.cl.createMultiBody(0, col, basePosition=centre,
                                    baseOrientation=_quat_from_z(n))
        if geom is not None:
            for c, h in geom['static_boxes']:
                col = self.cl.createCollisionShape(p.GEOM_BOX, halfExtents=list(h))
                self.cl.createMultiBody(0, col, basePosition=list(c))
        self.bodies = {}

    def sync(self, packed):
        for it in packed:
            if it.get('pos') is None:
                continue
            pos = (it['pos'][0] - self.ox, it['pos'][1], it['pos'][2])
            bid = self.bodies.get(it['index'])
            if bid is None:
                col = self.cl.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[it['length']/2, it['width']/2, it['height']/2])
                bid = self.cl.createMultiBody(it['mass'], col, basePosition=pos,
                                              baseOrientation=it['orn'])
                self.cl.changeDynamics(bid, -1, **_dyn(it))
                self.bodies[it['index']] = bid
            else:
                self.cl.resetBasePositionAndOrientation(bid, pos, it['orn'])
                self.cl.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])

    def settles(self, spec, pos, orn_quat, steps=300):
        col = self.cl.createCollisionShape(
            p.GEOM_BOX, halfExtents=[spec['length']/2, spec['width']/2, spec['height']/2])
        bid = self.cl.createMultiBody(spec['mass'], col, basePosition=pos, baseOrientation=orn_quat)
        self.cl.changeDynamics(bid, -1, **_dyn(spec))
        sid = self.cl.saveState()
        for _ in range(steps):
            self.cl.stepSimulation()
        fp, fo = self.cl.getBasePositionAndOrientation(bid)
        disp = math.dist(fp, pos)
        dot = min(1.0, abs(sum(a*b for a, b in zip(orn_quat, fo))))
        ang = math.degrees(2*math.acos(dot))
        self.cl.restoreState(sid)
        self.cl.removeState(sid)
        self.cl.removeBody(bid)
        return disp, ang


if __name__ == '__main__':
    from diag_stuck import run
    import packer, geometry
    from src.ground_handling.utils import ORNS
    nm = sys.argv[1] if len(sys.argv) > 1 else 'P00'
    cfg = json.load(open('scenes_pool.json'))[nm]
    cfg['containers']['container_list'] = cfg['containers']['container_list'][:1]
    cfg['camera']['num_containers'] = 1
    env, obs, last, steps = run(nm, cfg)
    cd = env.container_manager.get_item_info_in_containers()[0]
    t0 = time.perf_counter(); tw = Twin(cd); t_build = time.perf_counter()-t0
    t0 = time.perf_counter(); tw.sync(cd['packed_items']); t_sync = time.perf_counter()-t0
    it = env.stream_manager.visible_pool[0]
    spec = dict(index=it.index, length=it.length, width=it.width, height=it.height,
                mass=it.mass, lateralFriction=it.lateralFriction,
                is_prioritized=it.is_prioritized, is_soft=it.is_soft)
    cs = packer.ContainerState(cd)
    r = packer.best_placement(cs, spec, time.perf_counter()+20)
    print(f'{nm}: boxes={len(cd["packed_items"])} build={t_build*1000:.0f}ms sync={t_sync*1000:.0f}ms')
    if r:
        pos, orn, sc = r
        q = p.getQuaternionFromEuler(ORNS[orn])
        for steps_n in (300, 120, 60):
            t0 = time.perf_counter()
            d, a = tw.settles(spec, pos, q, steps_n)
            print(f'   settle({steps_n:3d}) = {(time.perf_counter()-t0)*1000:6.1f}ms  disp={d:.4f}m angle={a:.1f}deg')
    env.close()
