#!/bin/bash
# Construction windows on the public suite (diverse items), same load:
#   WA: windows (0,) frac 0.35  (= CON2 + gate, re-run for the noise level)
#   WB: windows (15,25) frac 0.5 (bounded per-step work, like the public phase 1)
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
C="--agent-dir $DEV --optimize-budget 150 --policy-budget 5.0 --set a:OFFLINE_CONSTRUCT=1"
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag WA_suite --jobs 7 $C > WA_suite.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag WB_suite --jobs 7 $C --set a:CONSTRUCT_PRESET=1 --set a:CONSTRUCT_FRAC=0.5 > WB_suite.log 2>&1 &
wait
echo ALLDONE
