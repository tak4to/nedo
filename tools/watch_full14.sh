#!/bin/bash
# Wait for the full14 CEM to finish, then verify it on unseen pools.
cd /home/takato/comp/nedo/tools
while pgrep -f 'cem.py .*full14' > /dev/null; do sleep 120; done
echo "=== CEM finished at $(date) ==="
grep -vE 'pybullet build|^argv' cem_full14.log | tail -20
echo
echo "=== verification ==="
../.venv/bin/python verify_cem.py --state cem_full14.json --prefix V14 2>&1 | grep -vE 'pybullet build|^argv'
echo "=== done at $(date) ==="
