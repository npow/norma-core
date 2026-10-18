#!/usr/bin/env python3
"""Run a policy through Station. Preview by default; --execute enables Space to move."""
import argparse
import asyncio
import base64
from dataclasses import dataclass, replace
import io
import json
import math
import queue
import threading
from pathlib import Path
import time
import urllib.request


class DiagnosticWriter:
    """Keep disk and terminal backpressure out of the motor-control loop."""
    def __init__(self, path):
        self.queue = queue.Queue(maxsize=2048)
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self.thread.start()

    def _run(self, path):
        try:
            with path.open("w", buffering=64 * 1024) as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        return
                    kind, value = item
                    if kind == "status":
                        print(value, flush=True)
                    else:
                        stream.write(value)
        except Exception as exc:
            self.error = exc

    def _put(self, item):
        if self.error is not None:
            raise RuntimeError(f"Diagnostic writer failed: {self.error}")
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            raise RuntimeError("Diagnostic output cannot keep up; stopped") from None

    def write(self, text):
        self._put(("data", text))

    def status(self, text):
        self._put(("status", text))

    def close(self):
        self._put(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Diagnostic output is blocked; log may be incomplete")
        if self.error is not None:
            raise RuntimeError(f"Diagnostic writer failed: {self.error}")


@dataclass
class TimedTrajectory:
    observed_at: float
    fps: float
    actions: list
    blend_started: float | None = None
    blend_seconds: float = 0.0
    position_offset: list | None = None
    velocity_offset: list | None = None

    def raw_action_at(self, now):
        coordinate = max(0, (now - self.observed_at) * self.fps)
        index = math.floor(coordinate + 1e-7)
        if index >= len(self.actions):
            raise RuntimeError("Action horizon expired while waiting for inference; stopped")
        fraction = max(0, coordinate - index)
        following = self.actions[min(index + 1, len(self.actions) - 1)]
        return [a + fraction * (b - a) for a, b in zip(self.actions[index], following, strict=True)]

    def velocity_at(self, now):
        coordinate = max(0, (now - self.observed_at) * self.fps)
        index = math.floor(coordinate + 1e-7)
        if index >= len(self.actions):
            raise RuntimeError("Cannot splice an expired action chunk")
        following = self.actions[min(index + 1, len(self.actions) - 1)]
        velocity = [(b - a) * self.fps for a, b in zip(self.actions[index], following, strict=True)]
        if self.blend_started is not None and 0 <= now - self.blend_started < self.blend_seconds:
            u = (now - self.blend_started) / self.blend_seconds
            dh00 = (6 * u * u - 6 * u) / self.blend_seconds
            dh10 = 3 * u * u - 4 * u + 1
            velocity = [v + dh00 * p + dh10 * dv for v, p, dv in zip(velocity, self.position_offset, self.velocity_offset, strict=True)]
        return velocity

    def splice_from(self, old, now, seconds):
        if seconds <= 0:
            return
        previous, previous_velocity = old.action_at(now), old.velocity_at(now)
        current, current_velocity = self.raw_action_at(now), self.velocity_at(now)
        remaining = self.observed_at + len(self.actions) / self.fps - now
        self.blend_started, self.blend_seconds = now, min(seconds, remaining)
        self.position_offset = [a - b for a, b in zip(previous, current, strict=True)]
        self.velocity_offset = [a - b for a, b in zip(previous_velocity, current_velocity, strict=True)]
        # Keep discrete grasp/release timing out of the arm-trajectory bridge.
        self.position_offset[-1] = self.velocity_offset[-1] = 0.0

    def action_at(self, now):
        action = self.raw_action_at(now)
        if self.blend_started is not None and 0 <= now - self.blend_started < self.blend_seconds:
            u = (now - self.blend_started) / self.blend_seconds
            h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
            action = [a + h00 * p + h10 * self.blend_seconds * dv for a, p, dv in zip(action, self.position_offset, self.velocity_offset, strict=True)]
        return action


def response_is_fresh(observed_at, now, fps, chunk_size, replan_steps):
    age = now - observed_at
    # Require a response within one replan interval and enough future actions
    # for another complete replan. Do not jump into the tail of an old chunk.
    return 0 <= age <= replan_steps / fps and age < (chunk_size - replan_steps) / fps


async def predict_async(url, observed_at, images, state, noise_seed=None):
    encode_started = time.monotonic()
    payload = await asyncio.to_thread(payload_for, images, state)
    encoded_at = time.monotonic()
    payload["noise_seed"] = noise_seed
    response = await asyncio.to_thread(request_json, url + "/predict", payload)
    response.setdefault("client_timings_ms", {}).update(
        observation_to_encode=1000 * (encode_started - observed_at),
        encode_and_dispatch=1000 * (encoded_at - encode_started),
        request_and_dispatch=1000 * (time.monotonic() - encoded_at),
    )
    return observed_at, response, 1000 * (time.monotonic() - observed_at)


# Policy traffic goes directly to the SSH loopback endpoint. macOS proxy
# autodiscovery is unnecessary here and can delay local requests.
_direct_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url, payload=None):
    started = time.monotonic()
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    encoded = time.monotonic()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _direct_http.open(request, timeout=10) as response:
        body = response.read()
    received = time.monotonic()
    result = json.loads(body)
    if payload is not None:
        result["client_timings_ms"] = {"serialize": 1000 * (encoded - started),
            "http_send_receive": 1000 * (received - encoded),
            "response_decode": 1000 * (time.monotonic() - received)}
    return result


def protected_commands(actions, motors, protection, config, max_delta):
    """Absolute calibrated actions -> raw goals, retaining Station's protection."""
    from mirror import compute_motor_command, get_steps_range, normalize_position
    if len(actions) != len(motors) or not all(math.isfinite(float(a)) for a in actions):
        raise ValueError("Invalid action dimensions or non-finite values")
    commands = []
    for i, (action, motor) in enumerate(zip(actions, motors, strict=True)):
        width = get_steps_range(motor.range_min, motor.range_max, config)
        if width <= 2 * config.safety_margin:
            raise ValueError(f"Motor {i + 1}: invalid calibrated arc")
        percent = float(action) if i == len(motors) - 1 else (float(action) + 100) / 2
        offset = min(width - config.safety_margin, max(config.safety_margin, percent / 100 * width))
        current = normalize_position(motor.present_position, motor.range_min, motor.range_max, config) / 100 * width
        offset = min(current + max_delta, max(current - max_delta, offset))
        offset = min(width - config.safety_margin, max(config.safety_margin, offset))
        raw = (motor.range_min + round(offset)) % config.max_steps
        # The learned action already contains the recorder's safety margin.
        # A virtual source in the follower's calibration avoids applying it twice.
        virtual = replace(motor, present_position=raw, error_status=0)
        command = compute_motor_command(i + 1, virtual, motor, protection.get(i + 1), replace(config, safety_margin=0))
        if command is not None:
            commands.append(command)
    return commands


def payload_for(images, state):
    from PIL import Image
    encoded = {}
    for key, pixels in images.items():
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
        encoded[key] = base64.b64encode(buffer.getvalue()).decode()
    return {"state": state.tolist(), "images": encoded}


async def run(args):
    from bootstrap import setup
    setup()
    from station_io import Station, vector
    from commands import send_motor_commands, set_torque
    from record import keyboard

    metadata = await asyncio.to_thread(request_json, args.policy_url + "/metadata")
    collection = metadata["collection"]
    leader, follower = collection["leader_bus"], collection["follower_bus"]
    cameras = collection["cameras"]
    fps = collection["fps"]
    station = await Station.connect(args.server)
    armed = False
    pending = None
    diagnostics = None
    try:
        await station.discover()
        station.configure_follower(follower, collection["motor_ids"])
        # Trajectory tracking needs finer updates than hand-guided teleoperation.
        station.config = replace(station.config, deadband=args.deadband_ticks)
        if list(station.motor_ids) != collection["motor_ids"]:
            raise RuntimeError("Motor order differs from training")

        def observe():
            _, motors, images, calibration = station.snapshot(None, follower, cameras, args.max_age)
            if calibration != {follower: collection["calibration_ranges"][follower]}:
                raise RuntimeError("Calibration differs from the training dataset")
            for key, image in images.items():
                if list(image.shape) != metadata["features"][key]["shape"]:
                    raise RuntimeError(f"{key}: camera resolution differs from training")
            return motors, images

        initial_motors, initial_images = observe()
        # Initialize the PNG encoder before the user starts a trial.
        await asyncio.to_thread(payload_for, initial_images, vector([m.present_position for m in initial_motors], initial_motors))
        print(f"Checkpoint: {metadata['checkpoint']}")
        print(f"{len(station.motor_ids)} motors; {fps} Hz; asynchronous replan every {args.chunk_steps} steps.")
        print("Space starts/pauses; q quits. Support the arm when torque is released." if args.execute else "PREVIEW: no motor writes. Space starts/pauses predictions; q quits.")
        running = False
        trajectory = None
        started = None
        last_status = 0.0
        last_request = float("-inf")
        last_control = None
        control_ticks = 0
        control_started = None
        inference_ms = None
        roundtrip_ms = None
        log_path = args.diagnostics_jsonl or (Path.home() / "benchmarks/white-lid-smolvla-20261017/rollouts" / f"{metadata.get('policy_type', 'policy')}-{time.time_ns()}.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = DiagnosticWriter(log_path)
        diagnostics.write(json.dumps({"event": "start", "checkpoint": metadata["checkpoint"], "backend": metadata.get("backend", "eager"), "execute": args.execute, "fps": fps, "replan_steps": args.chunk_steps, "deadband_ticks": args.deadband_ticks, "noise_seed": args.noise_seed}) + "\n")
        print(f"Diagnostics: {log_path}")
        deadline = time.monotonic()
        with keyboard() as keys:
            while True:
                key = keys.get_nowait() if not keys.empty() else None
                if key in ("q", "\x04"):
                    break
                if key == " ":
                    running = not running
                    trajectory = None
                    if pending is not None:
                        pending.cancel()
                        pending = None
                    if running:
                        observe()
                        started = time.monotonic()
                        last_request = float("-inf")
                        last_control = None
                        control_started = None
                        control_ticks = 0
                        diagnostics.status("WAITING FOR FRESH PREDICTION (torque not enabled yet)")
                    else:
                        if armed:
                            await station.disable(follower)
                            armed = False
                        print("PAUSED", flush=True)
                    deadline = time.monotonic()
                if not running:
                    await asyncio.sleep(0.025)
                    continue
                if trajectory is not None and time.monotonic() - started >= args.seconds:
                    if armed:
                        await station.disable(follower)
                        armed = False
                    running = False
                    trajectory = None
                    if pending is not None:
                        pending.cancel()
                        pending = None
                    print("Episode duration reached. Reset scene; Space starts another trial.", flush=True)
                    continue
                observe_started = time.monotonic()
                motors, images = observe()
                observe_ms = 1000 * (time.monotonic() - observe_started)
                now = time.monotonic()
                if pending is not None and pending.done():
                    observed_at, response, roundtrip_ms = pending.result()
                    pending = None
                    chunk = response["actions"]
                    if len(chunk) != metadata["chunk_size"] or any(len(row) != len(motors) or not all(math.isfinite(float(v)) for v in row) for row in chunk):
                        raise RuntimeError("Policy returned an invalid action chunk")
                    inference_ms = response.get("inference_ms")
                    fresh = response_is_fresh(observed_at, now, fps, len(chunk), args.chunk_steps)
                    diagnostics.write(json.dumps({"event": "inference", "time_s": now - started,
                        "accepted": fresh, "response_age_ms": 1000 * (now - observed_at),
                        "inference_ms": inference_ms, "roundtrip_ms": roundtrip_ms,
                        "client_timings_ms": response.get("client_timings_ms"),
                        "server_timings_ms": response.get("timings_ms")}) + "\n")
                    if not fresh:
                        last_request = float("-inf")
                        diagnostics.status(f"Discarded late prediction ({1000 * (now - observed_at):.0f} ms old); requesting fresh observation")
                    else:
                        candidate = TimedTrajectory(observed_at, fps, chunk)
                        candidate.action_at(now)
                        if trajectory is not None:
                            candidate.splice_from(trajectory, now, args.blend_seconds)
                        else:
                            # Enable motion only after accepting a current observation.
                            if args.execute:
                                armed = True
                                await set_torque(station.client, follower, station.motor_ids, enable=True)
                            started = now
                            diagnostics.status("RUNNING")
                        trajectory = candidate
                if pending is None and now - last_request >= args.chunk_steps / fps:
                    last_request = now
                    state = vector([m.present_position for m in motors], motors)
                    pending = asyncio.create_task(predict_async(args.policy_url, now, images, state, None if args.noise_seed == -1 else args.noise_seed))
                if trajectory is None:
                    await asyncio.sleep(0.005)
                    deadline = time.monotonic()
                    continue
                action = trajectory.action_at(now)
                commands = protected_commands(action, motors, station.protection, station.config, args.max_delta_ticks)
                send_started = time.monotonic()
                if args.execute:
                    await send_motor_commands(station.client, follower, commands)
                send_ms = 1000 * (time.monotonic() - send_started)
                current_state = vector([m.present_position for m in motors], motors)
                goals = [m.target_position for m in motors]
                for command in commands:
                    if command.goal is not None:
                        goals[command.motor_id - 1] = command.goal
                sent_state = vector(goals, motors)
                if control_started is None:
                    control_started = now
                control_ticks += 1
                hz = (control_ticks - 1) / max(now - control_started, 1e-6)
                diagnostics.write(json.dumps({"event": "control", "time_s": now - started, "interval_ms": None if last_control is None else 1000 * (now - last_control), "action_age_ms": 1000 * (now - trajectory.observed_at), "predicted_action": action, "measured_state": current_state.tolist(), "protected_goal": sent_state.tolist(), "currents": [m.current for m in motors], "observe_ms": observe_ms, "send_ms": send_ms, "commands": [{"motor_id": c.motor_id, "goal": c.goal, "speed": c.speed, "accel": c.accel} for c in commands]}) + "\n")
                last_control = now
                if time.monotonic() - last_status >= 1:
                    diagnostics.status(f"{now - started:4.1f}s {hz:4.1f}Hz infer={inference_ms or 0:.0f}ms rtt={roundtrip_ms or 0:.0f}ms grip(pred/sent/actual)={action[-1]:.1f}/{sent_state[-1]:.1f}/{current_state[-1]:.1f} action={[round(float(v), 1) for v in action]}")
                    last_status = time.monotonic()
                deadline += 1 / fps
                if deadline < time.monotonic():
                    # Do not play queued trajectory points late or in catch-up bursts.
                    deadline = time.monotonic() + 1 / fps
                await asyncio.sleep(max(0, deadline - time.monotonic()))
    finally:
        if pending is not None:
            pending.cancel()
        try:
            if armed:
                task = asyncio.create_task(station.disable(follower))
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
        finally:
            try:
                if diagnostics is not None:
                    diagnostics.close()
            finally:
                station.client.close_connection()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-url", default="http://127.0.0.1:8765")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--max-age", type=float, default=0.2)
    parser.add_argument("--max-delta-ticks", type=int, default=60)
    parser.add_argument("--diagnostics-jsonl", type=Path)
    parser.add_argument("--blend-seconds", type=float, default=0.15)
    parser.add_argument("--deadband-ticks", type=int, default=20)
    parser.add_argument("--noise-seed", type=int, default=-1, help="Reuse sampling noise across replans; -1 samples fresh noise")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.chunk_steps <= 20 or not 0 < args.max_age <= 1 or args.max_delta_ticks <= 0:
        parser.error("Use positive duration/delta, 1–20 chunk steps, and max-age in (0, 1]")
    if not math.isfinite(args.blend_seconds) or not 0 <= args.blend_seconds <= 0.5:
        parser.error("--blend-seconds must be in [0, 0.5]")
    if not -1 <= args.noise_seed < 2**63:
        parser.error("--noise-seed must be -1 or an integer in [0, 2**63)")
    if not 1 <= args.deadband_ticks <= 20:
        parser.error("--deadband-ticks must be in [1, 20]")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError, urllib.error.URLError) as exc:
        parser.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
