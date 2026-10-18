"""Start checkpoint inference through SSH, then run the local Station client."""
import os
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def main():
    here = Path(__file__).resolve().parent
    os.chdir(here)
    kind = os.environ.get("POLICY_TYPE", "act")
    if kind not in ("act", "smolvla"):
        raise SystemExit("POLICY_TYPE must be act or smolvla")
    step_key = "SMOLVLA_STEP" if kind == "smolvla" else "ACT_STEP"
    step = os.environ.get(step_key, "2000" if kind == "smolvla" else "5000")
    if not step.isascii() or not step.isdigit() or not 1 <= int(step) <= 20000:
        raise SystemExit(f"{step_key} must be an integer in [1, 20000]")
    root = "/home/npow/benchmarks/white-lid-smolvla-20261017"
    model_root = "smolvla-checkpoints" if kind == "smolvla" else "act-cleaned-20k/checkpoints"
    checkpoint = f"{root}/{model_root}/{int(step):06d}/pretrained_model"
    port = os.environ.get("POLICY_PORT", "8765")
    if not port.isascii() or not port.isdigit() or not 1024 <= int(port) <= 65535:
        raise SystemExit("POLICY_PORT must be an integer in [1024, 65535]")
    policy_url = f"http://127.0.0.1:{port}"
    subprocess.run(["ssh", "odin", f"test -f {checkpoint}/model.safetensors"], check=True)
    backend = os.environ.get("POLICY_BACKEND", "compile" if kind == "smolvla" else "eager")
    if backend not in ("eager", "compile"):
        raise SystemExit("POLICY_BACKEND must be eager or compile")
    subprocess.run(["scp", "-q", str(here / "policy_server.py"), str(here / "policy_service.py"), f"odin:{root}/"], check=True)
    command = ["env", "PYTHONPATH=/home/npow/code/lerobot/src:/home/npow/code/lerobot/.venv/lib/python3.13/site-packages",
               "/home/npow/.local/bin/uv", "run", "--no-project", "--python", f"{root}/env/bin/python",
               "python", f"{root}/policy_service.py", "--checkpoint", checkpoint, "--dataset", f"{root}/dataset-cleaned", "--backend", backend]
    server = subprocess.Popen(["ssh", "-T", "odin", shlex.join(command)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    server.stdin.write(json.dumps({"HF_TOKEN": os.environ.get("HF_TOKEN")}) + "\n")
    server.stdin.close()
    endpoint = {}
    ready = threading.Event()

    def show_server_output():
        for line in server.stdout:
            if line.startswith("ENDPOINT_READY "):
                endpoint.update(json.loads(line.removeprefix("ENDPOINT_READY ")))
                print(f"READY {endpoint['backend']} {checkpoint} on Odin:{endpoint['port']}", flush=True)
                ready.set()
            else:
                print(line, end="", flush=True)

    threading.Thread(target=show_server_output, daemon=True).start()
    tunnel = None
    try:
        for _ in range(610):
            if ready.wait(timeout=1):
                break
            if server.poll() is not None:
                raise RuntimeError("Inference service exited before becoming ready")
        else:
            raise RuntimeError("Inference service startup timed out")
        server.wait(timeout=5)
        if server.returncode:
            raise RuntimeError("Inference service startup failed")
        tunnel = subprocess.Popen(["ssh", "-N", "-T", "-o", "ExitOnForwardFailure=yes", "-L",
            f"{port}:127.0.0.1:{endpoint['port']}", "odin"])
        for _ in range(50):
            if tunnel.poll() is not None:
                raise RuntimeError("SSH tunnel exited; check whether POLICY_PORT is already in use")
            try:
                with urllib.request.urlopen(policy_url + "/metadata", timeout=1) as response:
                    metadata = json.load(response)
                if metadata["checkpoint"] != checkpoint or metadata.get("backend") != backend:
                    raise RuntimeError("Local port belongs to another policy endpoint")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(.1)
        else:
            raise RuntimeError("SSH tunnel did not become ready")
        print(f"WARM {backend} endpoint ready; worker stays loaded for 30 minutes after last request.", flush=True)
        return subprocess.call([sys.executable, str(here / "rollout_act.py"), "--policy-url", policy_url, *sys.argv[1:]])
    finally:
        for child in (tunnel, server):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
