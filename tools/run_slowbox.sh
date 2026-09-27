#!/bin/bash
# Is our wall-clock search more fragile on a slow box than their unit-budget one?
# Same 8 scenes, each solver run single-process under two conditions:
#   FAST: no pinning, no competing load
#   SLOW: pinned to 4 CPUs with 2 busy loops (what tools/preflight.py emulates
#         for the 4 vCPU judging box)
# Offline budget 60s for both so the whole matrix fits in about an hour.
cd /home/takato/comp/nedo/tools
SHIM=$(./make_pub_shim.sh)
run() { # tag, condition, extra args
  local tag=$1 cond=$2; shift 2
  if [ "$cond" = slow ]; then
    for i in 1 2; do taskset -c 0-3 bash -c 'while :; do :; done' & done
    local pids=$!
    MYSOLVER_OPTIMIZE_BUDGET=60 taskset -c 0-3 ../.venv/bin/python ab.py --scenes scenes_slow8.json \
      --tag $tag --jobs 1 --optimize-budget 60 --policy-budget 5.0 "$@" > $tag.log 2>&1
    kill %1 %2 2>/dev/null
  else
    MYSOLVER_OPTIMIZE_BUDGET=60 ../.venv/bin/python ab.py --scenes scenes_slow8.json \
      --tag $tag --jobs 1 --optimize-budget 60 --policy-budget 5.0 "$@" > $tag.log 2>&1
  fi
}
run SB_our_fast fast
run SB_our_slow slow
run SB_pub_fast fast --agent-dir $SHIM
run SB_pub_slow slow --agent-dir $SHIM
echo ALLDONE
