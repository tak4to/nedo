#!/bin/bash
# OFFLINE_CONSTRUCT=1 + CONSTRUCT_STRICT=1 (agents/dev) on: public suite (26), our optimize scenes
# with their own look_ahead (44), and the same 44 as official A (k=1).
# Baselines: OUR_suite, OUR_<set> (optimize scenes), UB_basek1 -- all agents/submit,
# optimize 150s, ~12-14 concurrent jobs. Same total load here (14).
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
C="--agent-dir $DEV --optimize-budget 150 --policy-budget 5.0 --set a:OFFLINE_CONSTRUCT=1"
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag CON2_suite --jobs 4 $C > CON2_suite.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_opt44.json   --tag CON2_opt44 --jobs 5 $C > CON2_opt44.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_UB_base.json --tag CON2_basek1 --jobs 5 $C > CON2_basek1.log 2>&1 &
wait
echo ALLDONE
