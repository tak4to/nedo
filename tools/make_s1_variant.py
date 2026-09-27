"""Build the S1 submission candidate: agents/submit + small-first seed orders
gated on item diversity, and nothing else.

docs/2026-09-23-戦略（多様な荷物の伸びしろ）.md §4 S1. The change is code, not a
constant, so tools/make_variant.py cannot make it; this applies three exact
text insertions to agents/submit/agent.py and refuses to run if any anchor is
missing or appears more than once.

    cd <repo root>
    .venv/bin/python tools/make_s1_variant.py

Output: agents/variants/s1_small_seeds/submit/ (for ab.py --agent-dir),
agents/variants/s1_small_seeds.zip (root dir `submit/`, like agents/submit.zip)
and agents/variants/s1_small_seeds.json (base hashes + unified diff).
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
TAG = 's1_small_seeds'
OUT_DIR = os.path.join(REPO, 'agents', 'variants', TAG)
SOURCES = ('agent.py', 'packer.py', 'geometry.py', 'twin.py')

CONSTANTS = """
# Small-first seed orders, on varied items only (2026-09-23). Every other
# seed the search starts from is big-first -- four descending sort keys,
# GRASP keys with positive weights -- while its objective is the placed
# count, so on items of many different sizes it never looked at orders
# that place the small ones first. On our 44 optimize scenes with random
# item dims (k=1) this took placed 1246 -> 1436 (packed% +7.32 se 1.18,
# fill -1.61 se 0.82); on the public repo's 26-scene suite 690 -> 830.
# On catalogue scenes (sizes close together) it buys no items and costs
# 2-3 fill, hence the gate: distinct shapes / items is 0.06-0.17 there
# (5-7 shapes) and 1.00 when every item has its own dimensions.
OFFLINE_SMALL_SEEDS = True
SMALL_SEEDS_MIN_DIVERSITY = 0.5
"""

DIVERSITY = """            diversity = len({tuple(sorted((s['length'], s['width'], s['height'])))
                             for s in specs}) / max(1, len(specs))
"""

SEEDS = """            if OFFLINE_SMALL_SEEDS and diversity >= SMALL_SEEDS_MIN_DIVERSITY:
                def sort_key_volume_asc(s):
                    return (1 if s['is_soft'] else 0, s['length'] * s['width'] * s['height'])

                def sort_key_maxdim_asc(s):
                    return (1 if s['is_soft'] else 0, max(s['length'], s['width'], s['height']),
                            s['length'] * s['width'] * s['height'])
                base_keys += [sort_key_volume_asc, sort_key_maxdim_asc]
"""

EDITS = [
    ("OFFLINE_SEARCH = 'lns'\n", CONSTANTS),
    ("            specs = [_item_spec(d) for d in item_list]\n", DIVERSITY),
    ("            base_keys = [sort_key_volume, sort_key_footprint, sort_key_height, sort_key_mass]\n", SEEDS),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def main():
    src = open(os.path.join(BASE_DIR, 'agent.py'), encoding='utf-8').read()
    new = src
    for anchor, insert in EDITS:
        n = new.count(anchor)
        if n != 1:
            sys.exit(f'anchor found {n} times, expected 1: {anchor!r}')
        new = new.replace(anchor, anchor + insert, 1)
    compile(new, 'agent.py', 'exec')

    pkg = os.path.join(OUT_DIR, 'submit')
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(pkg)
    for fn in SOURCES:
        shutil.copy2(os.path.join(BASE_DIR, fn), os.path.join(pkg, fn))
    with open(os.path.join(pkg, 'agent.py'), 'w', encoding='utf-8') as f:
        f.write(new)

    zpath = os.path.join(REPO, 'agents', 'variants', TAG + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('submit/', '')
        for fn in SOURCES:
            z.write(os.path.join(pkg, fn), f'submit/{fn}')

    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        'submit/agent.py', f'{TAG}/agent.py'))
    base = {fn: sha(open(os.path.join(BASE_DIR, fn), 'rb').read()) for fn in SOURCES}
    out = {fn: sha(open(os.path.join(pkg, fn), 'rb').read()) for fn in SOURCES}
    rev = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    rec = dict(tag=TAG, base_dir='agents/submit', git_head=rev, base_sha=base, out_sha=out,
               zip_sha=sha(open(zpath, 'rb').read()),
               changed=[fn for fn in SOURCES if base[fn] != out[fn]], diff=diff)
    json.dump(rec, open(os.path.join(REPO, 'agents', 'variants', TAG + '.json'), 'w'),
              indent=1, ensure_ascii=False)
    print(f'{zpath}  changed={rec["changed"]}  zip_sha={rec["zip_sha"]}')
    print(diff)


if __name__ == '__main__':
    main()
