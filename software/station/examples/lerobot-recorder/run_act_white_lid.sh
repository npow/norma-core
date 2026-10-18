#!/usr/bin/env bash
# One command: SSH inference on odin + local Station rollout. Preview by default.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python launch_act.py "$@"
