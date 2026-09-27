#!/bin/bash
# E1: balanced-support levels x TOP_TOL, 4 scene sets each, against agents/submit baselines.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
run() { # tag, then --set args
  local tag=$1; shift
  for s in pool ktest shelftest test; do
    f=scenes_$s.json
    [ -f ab_${tag}_$s.json ] && continue
    ../.venv/bin/python ab.py --scenes $f --tag ${tag}_$s --jobs 14 --optimize-budget 15 --agent-dir $DEV "$@" > E1_${tag}_$s.log 2>&1
  done
  ../.venv/bin/python e1_report.py $tag
}
A="--set p:SUPPORT_BALANCED=1 --set p:BAL_COVER=0.35 --set p:BAL_SPAN=0.40 --set p:BAL_CENTROID=0.25"
B="--set p:SUPPORT_BALANCED=1 --set p:BAL_COVER=0.45 --set p:BAL_SPAN=0.50 --set p:BAL_CENTROID=0.20"
C="--set p:SUPPORT_BALANCED=1 --set p:BAL_COVER=0.55 --set p:BAL_SPAN=0.60 --set p:BAL_CENTROID=0.15"
T="--set p:TOP_TOL=0.02"
run E1A $A
run E1A_t $A $T
run E1B $B
run E1B_t $B $T
run E1C $C
run E1C_t $C $T
echo ALLDONE
