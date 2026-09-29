"""Build the W_SLIVER submission candidate: agents/submit plus the
"no small offcut" penalty (docs/2026-09-23-現場の実務と再現・優先度.md §6.2),
and nothing else.

Like tools/make_s1_variant.py this is code, not a constant, so
tools/make_variant.py cannot make it. Two exact insertions into
agents/submit/packer.py, refused if an anchor is missing or repeated.

    cd <repo root>
    .venv/bin/python tools/make_sliver_variant.py

Output: agents/variants/sliver400/submit/ (for ab.py --agent-dir),
agents/variants/sliver400.zip (root dir `submit/`) and sliver400.json.
"""
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO, 'agents', 'submit')
DEV_DIR = os.path.join(REPO, 'agents', 'dev')
TAG = 'sliver400'
OUT_DIR = os.path.join(REPO, 'agents', 'variants', TAG)
SOURCES = ('agent.py', 'packer.py', 'geometry.py', 'twin.py')

WEIGHT = '400.0'
NOTE = """# Cutting-stock's "no small offcut" rule.
#
# At the jam state 23.6% of the container is air in columns narrower than
# the smallest item -- gaps between towers that nothing can ever use --
# while 89-96% of all candidate rejections are "too tall"
# (docs/2026-09-23-現場の実務と再現・優先度.md §6.2). W_WALL rewards landing
# flush against a wall and W_SEAM rewards lining up with the support, but
# nothing noticed that a placement leaves a 12cm strip beside it that no
# remaining item can ever enter.
#
# A gap is "flush" up to SLIVER_LO (the design clearance plus a little) and
# usable from SLIVER_HI (the catalogue's smallest footprint side plus
# clearance on both sides); in between it is waste, and the penalty is
# proportional to the width thrown away.
#
# Measured at this weight: on 6 dense scenes the void budget moves
# sliver 23.6% -> 20.8%, items 52.1% -> 54.2%, unreachable air 13.3% ->
# 10.8%. On 64 dense scenes (dev + holdout, paired) the placed count is
# flat -- packed% +0.20 (se 0.75) -- while fill gains +0.61 (se 0.61):
# the freed space goes to larger items rather than to more of them.
# 1500 and 5000 lose monotonically (-1.05 / -4.66 packed%).
"""

ANCHOR_FN = "def _side_contact_frac(cstate, cx, cy, hx, hy, bottom, top, fx, fy, fz):"
ANCHOR_APPLY = """    if W_WASTE and candidates:
        depths = _waste_depths(cstate, candidates)
        candidates = [(c[0] - W_WASTE * float(d),) + c[1:]
                      for c, d in zip(candidates, depths)]"""
APPLY = """
    if W_SLIVER and candidates:
        waste = _offcut_widths(cstate, candidates)
        candidates = [(c[0] - W_SLIVER * float(w),) + c[1:]
                      for c, w in zip(candidates, waste)]"""


def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def sliver_block():
    """The constants + helper, taken from agents/dev so the shipped code is
    byte-for-byte what was measured, minus the dev-only 'left off' note."""
    dev = open(os.path.join(DEV_DIR, 'packer.py'), encoding='utf-8').read()
    start = dev.index('# Cutting-stock')
    end = dev.index(ANCHOR_FN)
    block = dev[start:end]
    fn = block[block.index('def _offcut_widths'):]
    return NOTE + f"W_SLIVER = {WEIGHT}\n" + \
        "SLIVER_LO = GAP + 0.012      # at or below this the neighbour is effectively flush\n" + \
        "SLIVER_HI = 0.20 + 2 * GAP   # smallest catalogue side + clearance on both sides\n\n\n" + fn


def main():
    src = open(os.path.join(BASE_DIR, 'packer.py'), encoding='utf-8').read()
    new = src
    for anchor, insert in ((ANCHOR_FN, sliver_block()), (ANCHOR_APPLY, None)):
        if new.count(anchor) != 1:
            sys.exit(f'anchor found {new.count(anchor)} times, expected 1: {anchor[:60]!r}')
    new = new.replace(ANCHOR_FN, sliver_block() + ANCHOR_FN, 1)
    new = new.replace(ANCHOR_APPLY, ANCHOR_APPLY + APPLY, 1)
    compile(new, 'packer.py', 'exec')

    pkg = os.path.join(OUT_DIR, 'submit')
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(pkg)
    for fn in SOURCES:
        shutil.copy2(os.path.join(BASE_DIR, fn), os.path.join(pkg, fn))
    with open(os.path.join(pkg, 'packer.py'), 'w', encoding='utf-8') as f:
        f.write(new)

    zpath = os.path.join(REPO, 'agents', 'variants', TAG + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('submit/', '')
        for fn in SOURCES:
            z.write(os.path.join(pkg, fn), f'submit/{fn}')

    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        'submit/packer.py', f'{TAG}/packer.py'))
    base = {fn: sha(open(os.path.join(BASE_DIR, fn), 'rb').read()) for fn in SOURCES}
    out = {fn: sha(open(os.path.join(pkg, fn), 'rb').read()) for fn in SOURCES}
    rev = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    json.dump(dict(tag=TAG, base_dir='agents/submit', git_head=rev, base_sha=base, out_sha=out,
                   zip_sha=sha(open(zpath, 'rb').read()),
                   changed=[fn for fn in SOURCES if base[fn] != out[fn]], diff=diff),
              open(os.path.join(REPO, 'agents', 'variants', TAG + '.json'), 'w'),
              indent=1, ensure_ascii=False)
    print(f'{zpath}  changed={[fn for fn in SOURCES if base[fn] != out[fn]]}  zip_sha={sha(open(zpath, "rb").read())}')
    print(diff[:1500])


if __name__ == '__main__':
    main()
