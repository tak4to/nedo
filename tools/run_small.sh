#!/bin/bash
# Guard for OFFLINE_SMALL_SEEDS=1 on catalogue scenes + the public suite.
# Baselines: OUR_<set> (opt44), UB_basek1, OUR_suite -- agents/submit, 150s, ~14 jobs.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
C="--agent-dir $DEV --optimize-budget 150 --policy-budget 5.0 --set a:OFFLINE_SMALL_SEEDS=1"
../.venv/bin/python ab.py --scenes scenes_opt44.json    --tag SS_opt44  --jobs 5 $C > SS_opt44.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_UB_base.json  --tag SS_basek1 --jobs 5 $C > SS_basek1.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag SS_suite  --jobs 4 $C > SS_suite.log 2>&1 &
wait
echo ALLDONE
