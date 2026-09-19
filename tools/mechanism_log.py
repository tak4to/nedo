"""Per-item mechanism logger for stability (docs/2026-09-13-survey戦略.md P2).

The preliminary correlation study (same doc, §2) used only scene-level
aggregates -- mean displacement, cog_z, fill -- and found stability
uncorrelated with items placed, positively correlated with relative COM
height even controlling for fill, and that 61-68% of placed items move
>5cm under shake. That is consistent with several different mechanisms
(floor layer sliding, isolated items with no lateral support, an
unrestrained top layer, individual tip-overs) and scene-level numbers
cannot tell them apart.

This runs a handful of episodes with the current agent, and for every
placed item records what scores.stability_score's per-item output gives
(displacement, peak contact force, integrated kinetic energy) plus static
geometry facts computed independently: floor contact, wall contact count,
side-adjacent contact area with neighbours, headroom above the item, and
its position in the stack (final z rank within its container). It also
tags whether the item started at ground level (settled-in-place layer 0).

The output is one row per item across all sampled episodes, written to
mechanism_log.json, plus a summary that groups displacement by each
candidate mechanism -- read that first.

    ../.venv/bin/python mechanism_log.py --scenes scenes_off.json --n 12
"""
import argparse
import contextlib
import io
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _static_features(env, packer, geometry):
    """Floor/wall/neighbour-contact facts from the settled (pre-shake)
    layout, independent of scores.py's shake instrumentation."""
    rows = {}
    for c in env.container_manager.containers:
        g = packer.container_geometry({
            'index': c.index, 'length': c.length, 'width': c.width, 'height': c.height,
            'thickness': c.thickness, 'cut_x': c.cut_x, 'cut_y': c.cut_y,
            'center': c.center, 'points': c.points, 'n_vecs': c.n_vecs,
            'shelf': c.require_shelf, 'is_prioritized': c.is_prioritized,
            'packed_items': [],
        })
        boxes = []
        for it in c.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            hx, hy, hz = geometry.aabb_half_extents_from_quat(it.length, it.width, it.height, orn)
            boxes.append((it.index, pos, (hx, hy, hz)))
        top_z = max((p[2] + h[2] for _, p, h in boxes), default=g['floor_struct_z'])
        for idx, pos, half in boxes:
            cx, cy, cz = pos[0] - g['offset_x'], pos[1], pos[2]
            on_floor = abs((cz - half[2]) - g['floor_struct_z']) < 0.03
            walls = 0
            if cx - half[0] <= g['x_lo'] + 0.03:
                walls += 1
            if cx + half[0] >= g['x_hi'] - 0.03:
                walls += 1
            if cy + half[1] >= g['y_hi'] - 0.03:
                walls += 1
            if cy - half[1] <= g['y_lo'] + 0.03:
                walls += 1
            side_contact = 0.0
            for oidx, opos, ohalf in boxes:
                if oidx == idx:
                    continue
                ocx, ocy, ocz = opos[0] - g['offset_x'], opos[1], opos[2]
                oz_overlap = (min(cz + half[2], ocz + ohalf[2])
                              - max(cz - half[2], ocz - ohalf[2]))
                if oz_overlap < 0.02:
                    continue
                # side-adjacent on x: touching along y-extent, close in x
                if abs((cx + half[0]) - (ocx - ohalf[0])) < 0.03 or \
                   abs((cx - half[0]) - (ocx + ohalf[0])) < 0.03:
                    oy = min(cy + half[1], ocy + ohalf[1]) - max(cy - half[1], ocy - ohalf[1])
                    if oy > 0:
                        side_contact += oy * min(oz_overlap, 2 * half[2])
                if abs((cy + half[1]) - (ocy - ohalf[1])) < 0.03 or \
                   abs((cy - half[1]) - (ocy + ohalf[1])) < 0.03:
                    ox = min(cx + half[0], ocx + ohalf[0]) - max(cx - half[0], ocx - ohalf[0])
                    if ox > 0:
                        side_contact += ox * min(oz_overlap, 2 * half[2])
            rows[idx] = dict(on_floor=on_floor, n_walls=walls, side_contact=side_contact,
                             headroom=top_z - (cz + half[2]), rel_z=cz / g['height'],
                             container_top_z=top_z)
    return rows


def _one(job):
    name, cfg, agent_dir, seed = job
    os.environ['AGENT_DIR'] = agent_dir
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, agent_dir)
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    import packer
    import geometry
    import scores

    A.OPTIMIZE_TIME_BUDGET = 8.0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
        ag = A.Agent(module_path=agent_dir)
        env.reset_settings()
        ag.get_init_states(env.get_init_states())
        if cfg['agent'].get('optimize'):
            env.set_item_order(ag.optimize(env.get_info_for_optimization()))
        env.reset_item_stream()
        obs, _ = env.reset(seed=seed)
        term, steps = False, 0
        while not term and steps < 500:
            obs, _, term, _, _ = env.step(ag.policy(observation=obs))
            steps += 1
        static = _static_features(env, packer, geometry)
        out = scores.all_scores(env, per_item=True)
        env.close()

    rows = []
    for r in out['stability_per_item']:
        st = static.get(r['index'], {})
        rows.append(dict(scene=name, **r, **st))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--agent-dir', default=os.path.join(ROOT, 'agents', 'submit'))
    ap.add_argument('--out', default='mechanism_log.json')
    a = ap.parse_args()

    scenes = json.load(open(os.path.join(ROOT, 'tools', a.scenes)))
    names = list(scenes)[:a.n]

    from concurrent.futures import ProcessPoolExecutor
    jobs = [(nm, scenes[nm], a.agent_dir, 42 + i) for i, nm in enumerate(names)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        parts = list(ex.map(_one, jobs))
    rows = [r for part in parts for r in part]
    json.dump(rows, open(os.path.join(ROOT, 'tools', a.out), 'w'), indent=1)

    print(f"{len(rows)} items across {len(names)} scenes\n")
    print(f"share moved >5cm: {100 * sum(1 for r in rows if r['disp'] > 0.05) / len(rows):.0f}%\n")

    def bucket(label, keyfn, buckets):
        print(f"{label}:")
        for name, pred in buckets:
            sel = [r for r in rows if pred(r)]
            if not sel:
                print(f"    {name:20} n=0")
                continue
            print(f"    {name:20} n={len(sel):4d}  disp {statistics.mean(r['disp'] for r in sel):.3f}m  "
                  f"moved>5cm {100 * sum(1 for r in sel if r['disp'] > 0.05) / len(sel):.0f}%  "
                  f"force {statistics.mean(r['peak_force'] for r in sel):.0f}N")

    bucket("by floor contact", None, [
        ("on floor", lambda r: r.get('on_floor')),
        ("not on floor", lambda r: not r.get('on_floor', True)),
    ])
    bucket("by wall contacts", None, [
        (f"{k} walls", lambda r, k=k: r.get('n_walls') == k) for k in (0, 1, 2, 3, 4)
    ])
    bucket("by side-adjacent contact (isolation)", None, [
        ("isolated (<1cm^2)", lambda r: r.get('side_contact', 0) < 1e-4),
        ("some side contact", lambda r: r.get('side_contact', 0) >= 1e-4),
    ])
    rel_z = sorted(r.get('rel_z', 0) for r in rows)
    if rel_z:
        t1, t2 = rel_z[len(rel_z) // 3], rel_z[2 * len(rel_z) // 3]
        bucket("by height in container (tertiles)", None, [
            ("low", lambda r: r.get('rel_z', 0) <= t1),
            ("mid", lambda r: t1 < r.get('rel_z', 0) <= t2),
            ("high (near top)", lambda r: r.get('rel_z', 0) > t2),
        ])
    bucket("by headroom above item", None, [
        ("<5cm to skyline/lid", lambda r: r.get('headroom', 9) < 0.05),
        ("5-15cm", lambda r: 0.05 <= r.get('headroom', 9) < 0.15),
        (">=15cm", lambda r: r.get('headroom', 9) >= 0.15),
    ])
    bucket("by item class", None, [
        ("soft", lambda r: r.get('is_soft')),
        ("priority", lambda r: r.get('is_prioritized')),
        ("plain", lambda r: not r.get('is_soft') and not r.get('is_prioritized')),
    ])
    print("\nread the buckets above against the P2 hypothesis table:")
    print("  floor layer slides    -> 'on floor' should show high disp/force")
    print("  isolated items move   -> 'isolated' should show high disp")
    print("  top layer unrestrained-> 'high' height tertile / low headroom should show high disp")
    print("  individual tip-overs  -> disp should NOT track n_walls/side_contact broadly,")
    print("                           i.e. only a few outliers move a lot rather than a broad shift")


if __name__ == '__main__':
    main()
