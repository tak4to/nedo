#!/bin/bash
# Head-to-head: public solver (MasahiroTatsuta/baggage-packing @624d66a, their
# 61.092 main config via the scratchpad shim; local benchmark only, never
# submitted) vs agents/submit, both at production budgets, run concurrently
# at 7 jobs each so both arms see the same CPU load.
cd /home/takato/comp/nedo/tools
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
arm_pub() {
  for s in pool ktest shelftest test; do
    ../.venv/bin/python ab.py --scenes scenes_$s.json --tag PUB_$s --jobs 7 \
      --agent-dir $SHIM > PUB_$s.log 2>&1
  done
}
arm_our() {
  for s in pool ktest shelftest test; do
    ../.venv/bin/python ab.py --scenes scenes_$s.json --tag OUR_$s --jobs 7 \
      --optimize-budget 150 --policy-budget 5.0 > OUR_$s.log 2>&1
  done
}
arm_pub &
arm_our &
wait
echo ALLDONE
