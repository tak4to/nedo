#!/bin/bash
# Both solvers on the public repo's 26-scene suite (includes 6 prepacked scenes,
# which none of our 137 local scenes have). Same load: 7 jobs each, concurrent.
cd /home/takato/comp/nedo/tools
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag PUB_suite --jobs 7 --agent-dir $SHIM > PUB_suite.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_pubsuite.json --tag OUR_suite --jobs 7 --optimize-budget 150 --policy-budget 5.0 > OUR_suite.log 2>&1 &
wait
echo ALLDONE
