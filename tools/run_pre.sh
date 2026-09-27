#!/bin/bash
# (1) our solver with free choice on the public suite's optimize scenes (own code)
# (2) both solvers on our catalog test set + public prepacked layouts
cd /home/takato/comp/nedo/tools
SHIM=/tmp/claude-1000/-home-takato-comp-nedo/50c1fd44-26d0-4620-b4b3-9c997ee1cfb4/scratchpad/pub_shim
../.venv/bin/python ab.py --scenes scenes_pubsuite_choice.json --tag OUR_suite_choice --jobs 6 > OUR_suite_choice.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_test_pre.json --tag PUB_testpre --jobs 4 --agent-dir $SHIM > PUB_testpre.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_test_pre.json --tag OUR_testpre --jobs 4 --optimize-budget 150 --policy-budget 5.0 > OUR_testpre.log 2>&1 &
wait
echo ALLDONE
