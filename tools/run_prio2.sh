#!/bin/bash
# P3 weight sweep: how much W_PRIO_HIGH can we afford before packing suffers?
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
B="--scenes scenes_prio.json --jobs 4 --optimize-budget 15 --policy-budget 5.0 --agent-dir $DEV"
../.venv/bin/python ab.py $B --tag PR_h100 --set p:W_PRIO_HIGH=100 > PR_h100.log 2>&1 &
../.venv/bin/python ab.py $B --tag PR_h200 --set p:W_PRIO_HIGH=200 > PR_h200.log 2>&1 &
../.venv/bin/python ab.py $B --tag PR_h300 --set p:W_PRIO_HIGH=300 > PR_h300.log 2>&1 &
wait
echo ALLDONE
