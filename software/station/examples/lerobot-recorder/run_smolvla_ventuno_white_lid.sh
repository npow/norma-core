#!/usr/bin/env bash
set -euo pipefail
# Only terminal input crosses ADB; inference, cameras and motor control stay on Arduino.
ARDUINO_RUNNER="${SMOLVLA_ARDUINO_RUNNER:-${SMOLVLA_REMOTE_RUNNER:-/home/arduino/smolvla-white-lid/variants/192px-ab2-4/run_smolvla_white_lid.sh}}"
ARDUINO_COMMAND="$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$ARDUINO_RUNNER" "$@")"
exec adb -s "${SMOLVLA_ADB_SERIAL:-435318401}" shell -t "$ARDUINO_COMMAND"
