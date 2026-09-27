"""Pack an agent directory into a submission zip (root dir `submit/`, like
agents/submit.zip) and record what it is.

    cd <repo root>
    .venv/bin/python tools/pack_dir.py agents/submit --tag hotfix1 [--base agents/submit.zip]

Writes agents/variants/<tag>.zip and <tag>.json with the sha256 of the zip
and of every source file, plus a unified diff of each file against the
same file inside --base (default: the incumbent agents/submit.zip), so the
exact change a submission carries is on record. Runs the static preflight.
"""
import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'agents', 'variants')
FILES = ('agent.py', 'geometry.py', 'packer.py', 'twin.py')


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--base', default=os.path.join(REPO, 'agents', 'submit.zip'))
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f'{a.tag}.zip')
    if os.path.exists(out):
        os.remove(out)
    record = {'tag': a.tag, 'src': os.path.relpath(src, REPO), 'files': {}, 'diff_vs_base': {}}
    base = zipfile.ZipFile(a.base) if a.base and os.path.exists(a.base) else None
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(src, 'submit/')
        for fn in FILES:
            p = os.path.join(src, fn)
            data = open(p, 'rb').read()
            compile(data, p, 'exec')
            z.write(p, f'submit/{fn}')
            record['files'][fn] = _sha(data)
            if base is not None:
                old = base.read(f'submit/{fn}').decode('utf-8').splitlines(keepends=True)
                new = data.decode('utf-8').splitlines(keepends=True)
                d = ''.join(difflib.unified_diff(old, new, f'base/{fn}', f'{a.tag}/{fn}'))
                if d:
                    record['diff_vs_base'][fn] = d
    record['zip_sha256'] = _sha(open(out, 'rb').read())
    with open(os.path.join(OUT_DIR, f'{a.tag}.json'), 'w') as f:
        json.dump(record, f, indent=2)
    r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'preflight.py'),
                        '--zip', out, '--static-only'], capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(f'{os.path.relpath(out, REPO)}  sha256 {record["zip_sha256"][:12]}  '
          f'changed: {sorted(record["diff_vs_base"])}')
    if r.returncode != 0:
        sys.exit('static preflight failed')


if __name__ == '__main__':
    main()
