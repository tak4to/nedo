#!/bin/bash
# Characterise the shipped agent at the real operational density
# (50-60 bags per container, docs/2026-09-23-現場の実務と再現・優先度.md).
cd /home/takato/comp/nedo/tools
../.venv/bin/python ab.py --scenes scenes_dense.json     --tag DN_dev  --jobs 7 --optimize-budget 150 --policy-budget 5.0 > DN_dev.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_densetest.json --tag DN_test --jobs 7 --optimize-budget 150 --policy-budget 5.0 > DN_test.log 2>&1 &
wait
echo ALLDONE
