#!/bin/bash
# hotfix7 (soft bags in the offline order): hotfix6 vs S1 vs S2, run side by
# side at the same load, on every optimize scene (scenes_softA.json).
cd /home/takato/comp/nedo/tools
HF9=/home/takato/comp/nedo/agents/hf9
HF10=/home/takato/comp/nedo/agents/hf10
B="--scenes scenes_softA.json --jobs 4 --optimize-budget 60 --policy-budget 5.0"
../.venv/bin/python ab.py $B --tag SA_V6 --agent-dir $HF9                               > SA_V6.log 2>&1 &
../.venv/bin/python ab.py $B --tag SA_S1 --agent-dir $HF10                              > SA_S1.log 2>&1 &
../.venv/bin/python ab.py $B --tag SA_S2 --agent-dir $HF10 --set a:OFFLINE_ITEM_WORTH=10 > SA_S2.log 2>&1 &
wait
echo ALLDONE
