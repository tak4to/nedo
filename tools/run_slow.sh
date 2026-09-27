#!/bin/bash
# H-slow: public repo observed production policy ~10x slower than local (their
# fixed-work solver: 0.5-0.6s local vs 6.17s on the platform, read from the
# execution-time field SIGNATE displays). Emulate a slower machine by scaling
# every wall-clock budget: policy budget and MIN_CALL_BUDGET together.
# Offline stays 15s (= production 150s / 10, and the E1 setting).
# Three arms run concurrently at 5 jobs each so all see the same CPU load.
cd /home/takato/comp/nedo/tools
arm() { # tag policy_budget min_call
  for s in pool ktest shelftest test; do
    ../.venv/bin/python ab.py --scenes scenes_$s.json --tag ${1}_$s --jobs 5 \
      --optimize-budget 15 --policy-budget $2 --set p:MIN_CALL_BUDGET=$3 > ${1}_$s.log 2>&1
  done
}
arm SL10 5.0 0.8 &
arm SL03 1.5 0.24 &
arm SL01 0.5 0.08 &
wait
echo ALLDONE
