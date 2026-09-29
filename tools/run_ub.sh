#!/bin/bash
# Upper bound for item-selection construction (public-repo phase1 idea, own code only):
# optimize scenes as official A (optimize + look_ahead=1, LNS order at the full 150s)
# vs the same item lists with free choice among 40 (online selection, no optimize).
cd /home/takato/comp/nedo/tools
../.venv/bin/python ab.py --scenes scenes_UB_base.json   --tag UB_basek1 --jobs 6 --optimize-budget 150 > UB_basek1.log 2>&1 &
../.venv/bin/python ab.py --scenes scenes_UB_choice.json --tag UB_choice --jobs 6 > UB_choice.log 2>&1 &
wait
echo ALLDONE
