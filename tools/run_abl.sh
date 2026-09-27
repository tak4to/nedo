#!/bin/bash
# Where does the public solver's count lead on varied items come from?
# scenes_div44 (our 44 optimize scenes, random dims, k=1). 4 arms x 4 jobs.
#   OURs : agents/dev + OFFLINE_SMALL_SEEDS=1 (two ascending sort keys)
#   PUB ablations (env vars read by their modules; the shim only sets defaults):
#     oc0 ORDER_BY_COUNT=0 / pe0 PLAN_EXECUTOR=0 / lr0 LASTRESORT=0
# Baselines already measured at the same total load: DIV_our, DIV_pub.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_ours --jobs 4 --agent-dir $DEV --optimize-budget 150 --policy-budget 5.0 --set a:OFFLINE_SMALL_SEEDS=1 > DIV_ours.log 2>&1 &
MYSOLVER_ORDER_BY_COUNT=0 ../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_pub_oc0 --jobs 4 --agent-dir $SHIM > DIV_pub_oc0.log 2>&1 &
MYSOLVER_PLAN_EXECUTOR=0  ../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_pub_pe0 --jobs 4 --agent-dir $SHIM > DIV_pub_pe0.log 2>&1 &
MYSOLVER_LASTRESORT=0     ../.venv/bin/python ab.py --scenes scenes_div44.json --tag DIV_pub_lr0 --jobs 4 --agent-dir $SHIM > DIV_pub_lr0.log 2>&1 &
wait
echo ALLDONE
