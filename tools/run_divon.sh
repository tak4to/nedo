#!/bin/bash
# S4: online tasks (B: pool>1, C: k=1, no optimize) with random item dims.
cd /home/takato/comp/nedo/tools
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
../.venv/bin/python ab.py --scenes scenes_divon.json --tag DIVON_our --jobs 7 --policy-budget 5.0 > DIVON_our.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_divon.json --tag DIVON_pub --jobs 7 --agent-dir $SHIM > DIVON_pub.log 2>&1 &
wait
echo ALLDONE
