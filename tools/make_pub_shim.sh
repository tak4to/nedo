#!/bin/bash
# Re-create the local-benchmark shim for the public solver
# (MasahiroTatsuta/baggage-packing, pinned to the commit we measured).
#
# LOCAL COMPARISON ONLY. Their code never enters our submission: no LICENSE
# file means all rights reserved, and 他参加者のコードは流用しない
# (docs/手荷物積付コンペ top10 到達戦略 §10-2). Everything lands in the
# scratchpad, which /tmp clears on reboot -- run this again when it does.
#
#   tools/make_pub_shim.sh [scratchpad_dir]   # prints the --agent-dir to use
set -eu
SP="${1:-$(ls -d /tmp/claude-1000/-home-takato-comp-nedo/*/scratchpad 2>/dev/null | head -1)}"
REV=624d66a
[ -d "$SP/bp" ] || git clone -q https://github.com/MasahiroTatsuta/baggage-packing.git "$SP/bp"
git -C "$SP/bp" checkout -q $REV
mkdir -p "$SP/pub_shim"
cat > "$SP/pub_shim/agent.py" <<PY
"""Loads the public solver as a package so tools/harness.py can drive it.
Env vars reproduce their 61.092 main config (their HANDOVER.md section 8)."""
import os, sys
_BP = '$SP/bp/agents'
if _BP not in sys.path:
    sys.path.insert(0, _BP)
os.environ.setdefault('MYSOLVER_LASTRESORT', '1')
os.environ.setdefault('MYSOLVER_ORDER_BY_COUNT', '1')
os.environ.setdefault('MYSOLVER_OPTIMIZE_BUDGET', '135')
from mysolver.agent import Agent  # noqa: E402,F401
POLICY_TIME_BUDGET = None
OPTIMIZE_TIME_BUDGET = None
PY
# ab.py imports geometry/packer from the agent dir; the public solver keeps
# its own inside the package, so these stubs just satisfy the import.
: > "$SP/pub_shim/geometry.py"
: > "$SP/pub_shim/packer.py"
echo "$SP/pub_shim"
