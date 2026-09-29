#!/bin/bash
# Budget-only control for the construction seed: agents/submit with the LNS
# budget it would have left after CONSTRUCT_FRAC=0.35 (150*0.65 = 97.5s).
cd /home/takato/comp/nedo/tools
../.venv/bin/python ab.py --scenes scenes_opt44.json   --tag BUD_opt44  --jobs 7 --optimize-budget 97.5 --policy-budget 5.0 > BUD_opt44.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_UB_base.json --tag BUD_basek1 --jobs 7 --optimize-budget 97.5 --policy-budget 5.0 > BUD_basek1.log 2>&1 &
wait
echo ALLDONE
