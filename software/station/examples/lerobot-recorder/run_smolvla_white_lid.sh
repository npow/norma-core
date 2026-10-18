#!/usr/bin/env bash
# SmolVLA inference on odin + the same local Station rollout controller.
set -euo pipefail
cd "$(dirname "$0")"
export POLICY_TYPE=smolvla
exec uv run python launch_act.py "$@"
