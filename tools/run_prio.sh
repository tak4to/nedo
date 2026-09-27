#!/bin/bash
# P3: priority baggage door-side / high (docs/2026-09-23-現場の実務と再現・優先度.md).
# 123 scenes that contain priority items, 4 arms at the same load.
cd /home/takato/comp/nedo/tools
DEV=/home/takato/comp/nedo/agents/dev
B="--scenes scenes_prio.json --jobs 3 --optimize-budget 15 --policy-budget 5.0"
../.venv/bin/python ab.py $B --tag PR_base                                   > PR_base.log 2>&1 &
../.venv/bin/python ab.py $B --tag PR_door --agent-dir $DEV --set p:W_PRIO_DOOR=326.87 > PR_door.log 2>&1 &
../.venv/bin/python ab.py $B --tag PR_high --agent-dir $DEV --set p:W_PRIO_HIGH=415.8  > PR_high.log 2>&1 &
../.venv/bin/python ab.py $B --tag PR_both --agent-dir $DEV --set p:W_PRIO_DOOR=326.87 --set p:W_PRIO_HIGH=415.8 > PR_both.log 2>&1 &
wait
echo ALLDONE
