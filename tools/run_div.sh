#!/bin/bash
# Held-out check of the construction seed on OUR OWN diverse-item scenes (not the
# public suite): our 44 optimize scenes with random item dims, official A (k=1).
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_our --jobs 5 --optimize-budget 150 --policy-budget 5.0 > DIV_our.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_wb --jobs 5 --agent-dir $DEV --optimize-budget 150 --policy-budget 5.0 \
  --set a:OFFLINE_CONSTRUCT=1 --set a:CONSTRUCT_PRESET=1 --set a:CONSTRUCT_FRAC=0.5 > DIV_wb.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_pub --jobs 5 --agent-dir $SHIM > DIV_pub.log 2>&1 &
wait
echo ALLDONE
