"""Ensure a checkpoint-specific, warmed localhost worker exists on Odin."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from policy_server import load_credentials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backend", choices=("compile", "eager"), required=True)
    args = parser.parse_args()
    load_credentials(sys.stdin.buffer)
    server = Path(__file__).with_name("policy_server.py")
    identity = hashlib.sha256(server.read_bytes() + json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:12]
    port = 20000 + int(identity, 16) % 30000
    directory = server.parent / "serving" / identity
    directory.mkdir(parents=True, exist_ok=True)
    expected = {"checkpoint": args.checkpoint, "backend": args.backend}
    def health():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metadata", timeout=2) as response:
                metadata = json.load(response)
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Port {port} is occupied by another checkpoint")
            return metadata
        except (OSError, TimeoutError):
            return None
    with (directory / "start.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = health()
        if metadata is None:
            # Only our recorded worker can be reused; never stop unrelated services.
            pidfile = directory / "pid"
            running = False
            if pidfile.exists():
                try:
                    pid = int(pidfile.read_text())
                    command = Path(f"/proc/{pid}/cmdline").read_bytes()
                    running = str(server).encode() in command and args.checkpoint.encode() in command
                except (OSError, ValueError):
                    pass
            if not running:
                with (directory / "server.log").open("ab") as log:
                    child = subprocess.Popen([sys.executable, "-u", str(server), "--checkpoint", args.checkpoint,
                        "--dataset", args.dataset, "--backend", args.backend, "--port", str(port)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                pidfile.write_text(str(child.pid))
            print(f"Warming {args.backend} endpoint; log: {directory}/server.log", flush=True)
            deadline = time.monotonic() + 600
            while metadata is None and time.monotonic() < deadline:
                time.sleep(1)
                metadata = health()
                if metadata is None and not Path(f"/proc/{pidfile.read_text().strip()}").exists():
                    raise RuntimeError(f"Worker exited; inspect {directory}/server.log")
            if metadata is None:
                raise RuntimeError(f"Endpoint startup exceeded 600 seconds; inspect {directory}/server.log")
        print("ENDPOINT_READY " + json.dumps({"port": port, **metadata}), flush=True)


if __name__ == "__main__":
    main()
