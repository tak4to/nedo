"""Build a submission zip that differs from agents/submit by module constants.

For the submission-slot sweeps in docs/手荷物積付コンペ top10 到達戦略 §7 A-4:
one change per submission, several levels a day, and the local A/B cannot
judge them (se ~1 against effects of ~0.3), so what has to be exact is that
the zip differs from the incumbent by *only* the intended constants.

    cd <repo root>
    .venv/bin/python tools/make_variant.py SUPPORT_CENTROID_TOL=0.55
    .venv/bin/python tools/make_variant.py SUPPORT_MIN_COVER=0.55 --tag cover055
    .venv/bin/python tools/make_variant.py --list-sweep          # the planned A-4 grid

Output: agents/variants/<tag>.zip (root dir `submit/`, like agents/submit.zip)
plus <tag>.json recording base hashes and the exact source lines changed.
The edit is a top-level `NAME = value` line rewrite; it refuses to run if the
name is missing, defined twice, or the result does not compile.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO, 'agents', 'submit')
OUT_DIR = os.path.join(REPO, 'agents', 'variants')
SOURCES = ('agent.py', 'packer.py', 'geometry.py', 'twin.py')

# One variable per submission. Both knobs are only measured on the tightening
# side locally (tools/doe.py S3: MIN_COVER 0.63->0.72 = -1.10 (t=3.4),
# CENTROID_TOL 0.35->0.45 = +0.07 (t=0.4)); the loosening side is untested,
# and the doc's reference for this knob is a +3.54 from loosening it.
SWEEP = [
    ('SUPPORT_CENTROID_TOL', v) for v in (0.45, 0.55, 0.70)
] + [
    ('SUPPORT_MIN_COVER', v) for v in (0.55, 0.50)
]


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def apply_constants(work_dir, changes):
    """Rewrite `NAME = value` in exactly one source file per name."""
    record = []
    for name, value in changes:
        pat = re.compile(rf'^({re.escape(name)}\s*=\s*)([^#\n]+?)([ \t]*(?:#.*)?)$', re.M)
        hits = []
        for fn in SOURCES:
            p = os.path.join(work_dir, fn)
            if os.path.exists(p):
                src = open(p, encoding='utf-8').read()
                hits += [(fn, m) for m in pat.finditer(src)]
        if len(hits) != 1:
            sys.exit(f'{name}: expected exactly one top-level definition, found {len(hits)}')
        fn, m = hits[0]
        p = os.path.join(work_dir, fn)
        src = open(p, encoding='utf-8').read()
        new = m.group(1) + repr(value) + m.group(3)
        src = src[:m.start()] + new + src[m.end():]
        compile(src, p, 'exec')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(src)
        record.append({'name': name, 'file': fn, 'old': m.group(0), 'new': new})
        print(f'  {fn}: {m.group(0).strip()}  ->  {new.strip()}')
    return record


def build(changes, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, 'submit')
        shutil.copytree(BASE_DIR, work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        base_hashes = {fn: _sha(open(os.path.join(work, fn), 'rb').read())
                       for fn in os.listdir(work) if fn.endswith('.py')}
        record = apply_constants(work, changes)
        out = os.path.join(OUT_DIR, f'{tag}.zip')
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(work, 'submit/')
            for fn in sorted(os.listdir(work)):
                z.write(os.path.join(work, fn), f'submit/{fn}')
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump({'tag': tag, 'base': base_hashes, 'changes': record}, f, indent=2)
    return out


def parse(spec):
    name, _, raw = spec.partition('=')
    return name.strip(), json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('changes', nargs='*', help='NAME=VALUE (VALUE is a JSON literal)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--list-sweep', action='store_true', help='build the whole planned A-4 grid')
    args = ap.parse_args()

    jobs = []
    if args.list_sweep:
        jobs = [([(n, v)], f'{n.lower()}_{str(v).replace(".", "")}') for n, v in SWEEP]
    elif args.changes:
        ch = [parse(c) for c in args.changes]
        jobs = [(ch, args.tag or '_'.join(f'{n.lower()}_{str(v).replace(".", "")}' for n, v in ch))]
    else:
        ap.error('give NAME=VALUE or --list-sweep')

    for ch, tag in jobs:
        print(f'== {tag}')
        out = build(ch, tag)
        r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'preflight.py'),
                            '--zip', out, '--static-only'], capture_output=True, text=True)
        # zip differs from agents/submit by construction; only hard failures matter here
        if r.returncode != 0:
            print(r.stdout)
            sys.exit(f'{tag}: static preflight failed')
        print(f'  -> {os.path.relpath(out, REPO)}')


if __name__ == '__main__':
    main()
