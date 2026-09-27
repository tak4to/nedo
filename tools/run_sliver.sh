#!/bin/bash
# W_SLIVER sweep on the dense dev set (50-60 bags/container), base re-run under
# the same load so the comparison is paired and conditions match.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
B="--scenes scenes_dense.json --jobs 3 --optimize-budget 15 --policy-budget 5.0"
../.venv/bin/python ab.py $B --tag SV_base                                  > SV_base.log 2>&1 &
../.venv/bin/python ab.py $B --tag SV_400  --agent-dir $DEV --set p:W_SLIVER=400  > SV_400.log 2>&1 &
../.venv/bin/python ab.py $B --tag SV_1500 --agent-dir $DEV --set p:W_SLIVER=1500 > SV_1500.log 2>&1 &
../.venv/bin/python ab.py $B --tag SV_5000 --agent-dir $DEV --set p:W_SLIVER=5000 > SV_5000.log 2>&1 &
wait
echo ALLDONE
