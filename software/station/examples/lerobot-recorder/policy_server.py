#!/usr/bin/env python3
"""Load a LeRobot checkpoint on odin; serve action chunks over an SSH tunnel."""
import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import time


def load_credentials(stream):
    """Read one bounded SSH credential header before importing Hugging Face."""
    header = stream.readline(8193)
    if len(header) > 8192 or not header.endswith(b"\n"):
        raise ValueError("Invalid credential header")
    credentials = json.loads(header)
    if not isinstance(credentials, dict) or set(credentials) != {"HF_TOKEN"}:
        raise ValueError("Expected only HF_TOKEN in credential header")
    token = credentials["HF_TOKEN"]
    if token is not None and not isinstance(token, str):
        raise ValueError("Invalid HF_TOKEN type")
    if token:
        os.environ["HF_TOKEN"] = token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("eager", "compile"), default="eager")
    parser.add_argument("--idle-timeout", type=float, default=1800)
    parser.add_argument("--watch-stdin", action="store_true")
    parser.add_argument("--credentials-stdin", action="store_true")
    args = parser.parse_args()
    if args.credentials_stdin:
        load_credentials(sys.stdin.buffer)
    import numpy as np
    from PIL import Image
    import torch
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if args.backend == "compile":
        if not args.device.startswith("cuda"):
            raise ValueError("Compiled serving requires CUDA")
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    checkpoint = str(args.checkpoint.expanduser().resolve())
    kind = json.loads((Path(checkpoint) / "config.json").read_text())["type"]
    if kind == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = SmolVLAConfig.from_pretrained(checkpoint)
        # Modal's absolute backbone-cache path is not portable. The complete
        # fine-tuned weights are in this checkpoint; only the backbone config
        # and processor must be read from the locally cached public model.
        config.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        config.load_vlm_weights = False
        config.device = args.device
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
    elif kind == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(checkpoint, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint type: {kind}")
    policy = policy.to(args.device).eval()
    if args.backend == "compile":
        if kind != "smolvla":
            raise ValueError("Compiled backend is currently validated only for SmolVLA")
        policy.model.sample_actions = torch.compile(policy.model.sample_actions, mode="reduce-overhead")
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    def noise_arguments(seed):
        if kind != "smolvla" or seed is None:
            return {}
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("noise_seed must be null or an integer in [0, 2**63)")
        generator = None if seed is None else torch.Generator(device=args.device).manual_seed(seed)
        return {"noise": torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim),
                                      device=args.device, generator=generator)}

    collection = json.loads((args.dataset / "meta/collection.json").read_text())
    info = json.loads((args.dataset / "meta/info.json").read_text())
    metadata = {"checkpoint": checkpoint, "policy_type": kind, "collection": collection,
                "features": info["features"], "action_dim": policy.config.output_features["action"].shape[0],
                "chunk_size": policy.config.chunk_size, "backend": args.backend, "pid": os.getpid(), "supports_noise_seed": True}

    # Finish compilation and CUDA graph warmup before accepting observations.
    warmup = {"observation.state": torch.zeros(metadata["action_dim"]), "task": collection["task"]}
    warmup.update({key: torch.zeros(tuple(feature.shape)) for key, feature in policy.config.image_features.items()})
    with torch.inference_mode(), torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        for _ in range(3):
            if args.backend == "compile":
                torch.compiler.cudagraph_mark_step_begin()
            warm_actions = post(policy.predict_action_chunk(pre(warmup), **noise_arguments(None)).clone())
    if not torch.isfinite(warm_actions).all():
        raise RuntimeError("Checkpoint warm-up returned non-finite actions")

    last_activity = [time.monotonic()]

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            last_activity[0] = time.monotonic()
            self.respond(200 if self.path == "/metadata" else 404,
                         metadata if self.path == "/metadata" else {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.respond(404, {"error": "not found"})
            try:
                last_activity[0] = time.monotonic()
                request_started = time.perf_counter()
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("Expected a JSON observation smaller than 8 MiB")
                request = json.loads(self.rfile.read(length))
                state = np.asarray(request["state"], dtype=np.float32)
                if state.shape != (metadata["action_dim"],) or not np.isfinite(state).all():
                    raise ValueError("State dimension or values differ from the checkpoint")
                batch = {"observation.state": torch.from_numpy(state), "task": collection["task"]}
                for key, feature in policy.config.image_features.items():
                    with Image.open(io.BytesIO(base64.b64decode(request["images"][key], validate=True))) as image:
                        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                    expected = (feature.shape[1], feature.shape[2], feature.shape[0])
                    if pixels.shape != expected:
                        raise ValueError(f"{key}: expected {expected}, got {pixels.shape}")
                    batch[key] = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
                started = time.monotonic()
                with torch.inference_mode(), torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    processed = pre(batch)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    noise_kwargs = noise_arguments(request.get("noise_seed"))
                    model_started = time.perf_counter()
                    if args.backend == "compile":
                        torch.compiler.cudagraph_mark_step_begin()
                    normalized_actions = policy.predict_action_chunk(processed, **noise_kwargs).clone()
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    model_finished = time.perf_counter()
                    actions = post(normalized_actions).float().cpu().numpy()[0]
                if actions.shape != (policy.config.chunk_size, metadata["action_dim"]) or not np.isfinite(actions).all():
                    raise ValueError("Model returned invalid actions")
                self.respond(200, {"actions": actions.tolist(), "inference_ms": 1000 * (time.monotonic() - started),
                                   "timings_ms": {"read_decode": 1000 * (started - request_started),
                                                  "preprocess": 1000 * (model_started - started),
                                                  "model": 1000 * (model_finished - model_started),
                                                  "postprocess": 1000 * (time.perf_counter() - model_finished),
                                                  "server_total": 1000 * (time.perf_counter() - request_started)}})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": str(exc)})

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    def expire_when_idle():
        while True:
            time.sleep(30)
            if time.monotonic() - last_activity[0] > args.idle_timeout:
                server.shutdown()
                return
    threading.Thread(target=expire_when_idle, daemon=True).start()
    if args.watch_stdin:
        def stop_on_disconnect():
            sys.stdin.buffer.read()
            server.shutdown()
        threading.Thread(target=stop_on_disconnect, daemon=True).start()
    print(f"READY {checkpoint} on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
