#!/bin/bash
# W_SLIVER 200/400 on both dense sets (n=64), base re-run at the same load.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
for s in dense densetest; do
  B="--scenes scenes_$s.json --jobs 4 --optimize-budget 15 --policy-budget 5.0"
  ../.venv/bin/python ab.py $B --tag S2_base_$s                                 > S2_base_$s.log 2>&1 &
  ../.venv/bin/python ab.py $B --tag S2_200_$s --agent-dir $DEV --set p:W_SLIVER=200 > S2_200_$s.log 2>&1 &
  ../.venv/bin/python ab.py $B --tag S2_400_$s --agent-dir $DEV --set p:W_SLIVER=400 > S2_400_$s.log 2>&1 &
  wait
done
echo ALLDONE
