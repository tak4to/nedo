"""Calibrate scores.FORCE_SCALE / ENERGY_SCALE against the current agent.

scores.stability_score() computes three components the official evaluation
page names (docs/comp_determin.md): displacement after settling, force
during the shake, and kinetic energy during the shake. The pre-existing
displacement score used the convention "pick the constant so a typical
value scores ~37" (exp(-1)); this script measures "typical" for force and
energy the same way, over a handful of scenes with whatever agent is
currently on sys.path.

Run this again whenever the agent changes enough that force/energy might
have drifted, and copy the printed constants into scores.py by hand (kept
manual so a drifting agent can't silently rewrite its own scoring proxy).

    ../.venv/bin/python calib_stability.py --scenes scenes_off.json --n 10
"""
import argparse
import contextlib
import io
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='scenes_off.json')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--agent-dir', default=os.path.join(ROOT, 'agents', 'submit'))
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, 'src'))
    sys.path.insert(0, a.agent_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ground_handling.env import GroundHandlingEnv
    import agent as A
    import scores

    A.OPTIMIZE_TIME_BUDGET = 8.0
    cfgs = json.load(open(os.path.join(ROOT, 'tools', a.scenes)))
    names = list(cfgs)[:a.n]
    forces, energies, disps = [], [], []
    for nm in names:
        cfg = cfgs[nm]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = GroundHandlingEnv(config=cfg, verbose=False, render_mode=None)
            ag = A.Agent(module_path=a.agent_dir)
            env.reset_settings()
            ag.get_init_states(env.get_init_states())
            if cfg['agent'].get('optimize'):
                env.set_item_order(ag.optimize(env.get_info_for_optimization()))
            env.reset_item_stream()
            obs, _ = env.reset(seed=42)
            term, steps = False, 0
            while not term and steps < 500:
                obs, _, term, _, _ = env.step(ag.policy(observation=obs))
                steps += 1
            out = scores.all_scores(env)
            env.close()
        forces.append(out['shake_mean_peak_force'])
        energies.append(out['shake_mean_energy'])
        disps.append(out['shake_mean_disp'])
        print(f"{nm}: force={out['shake_mean_peak_force']:8.1f}N  "
              f"energy={out['shake_mean_energy']:7.4f}J  "
              f"disp={out['shake_mean_disp']:.3f}m")

    print(f"\n{'':10} {'mean':>10} {'median':>10} {'min':>10} {'max':>10}")
    for label, v in (('force (N)', forces), ('energy (J)', energies), ('disp (m)', disps)):
        print(f"{label:10} {statistics.mean(v):10.4f} {statistics.median(v):10.4f} "
              f"{min(v):10.4f} {max(v):10.4f}")
    print(f"\nsuggested constants (median -> score 37):")
    print(f"  FORCE_SCALE = {statistics.median(forces):.2f}")
    print(f"  ENERGY_SCALE = {statistics.median(energies):.4f}")


if __name__ == '__main__':
    main()
