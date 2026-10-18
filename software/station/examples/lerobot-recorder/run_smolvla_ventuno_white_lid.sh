#!/usr/bin/env bash
set -euo pipefail
# Only terminal input crosses ADB; inference, cameras and motor control stay on Arduino.
REMOTE_COMMAND="$(python3 -c 'import shlex, sys; print(shlex.join(["/home/arduino/smolvla-white-lid/run_smolvla_white_lid.sh", *sys.argv[1:]]))' "$@")"
exec adb -s "${SMOLVLA_ADB_SERIAL:-435318401}" shell -t "$REMOTE_COMMAND"
