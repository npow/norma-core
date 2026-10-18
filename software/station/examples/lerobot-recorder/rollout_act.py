#!/usr/bin/env python3
"""Run and record a Station policy, with optional leader mirroring and human takeover."""
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
    import numpy as np
    from station_io import Station, vector
    from commands import send_motor_commands, set_torque
    from record import keyboard, features_for
    from intervention import GrabDetector, LeaderMirror, RolloutRecorder, mirror_targets
    from mirror import ProtectionTable
    from state import find_bus

    recording = args.execute and not args.no_record
    if recording:
        # Load the ML/video stack before subscribing to live motor telemetry.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    metadata = await asyncio.to_thread(request_json, args.policy_url + "/metadata")
    collection = metadata["collection"]
    leader, follower = collection["leader_bus"], collection["follower_bus"]
    cameras, fps = collection["cameras"], collection["fps"]
    if not 0 < args.chunk_steps < metadata["chunk_size"]:
        raise ValueError("Replan interval must be smaller than the policy action horizon")
    if args.teleop_takeover and leader == follower:
        raise ValueError("Leader and follower must use different buses")
    station = await Station.connect(args.server)
    armed = False
    pending = diagnostics = recorder = mirror = None
    episode_open = False
    running = False
    mode = "policy"
    detector = GrabDetector(args.grab_error_ticks, args.grab_current, args.grab_seconds)
    try:
        await station.discover()
        if args.teleop_takeover:
            station.configure(leader, follower)
        else:
            station.configure_follower(follower, collection["motor_ids"])
        station.config = replace(station.config, deadband=args.deadband_ticks)
        if list(station.motor_ids) != collection["motor_ids"]:
            raise RuntimeError("Motor order differs from training")

        def observe():
            source, motors, images, calibration = station.snapshot(
                leader if args.teleop_takeover else None, follower, cameras, args.max_age)
            expected = {serial: collection["calibration_ranges"][serial] for serial in calibration}
            if calibration != expected:
                raise RuntimeError("Calibration differs from the training dataset")
            for key, image in images.items():
                if list(image.shape) != metadata["features"][key]["shape"]:
                    raise RuntimeError(f"{key}: camera resolution differs from training")
            return source, motors, images

        source, initial_motors, initial_images = observe()
        await asyncio.to_thread(payload_for, initial_images,
            vector([m.present_position for m in initial_motors], initial_motors))
        if args.teleop_takeover:
            mirror = LeaderMirror(station, leader, args.leader_torque_limit, args.max_delta_ticks)
        if recording:
            features = features_for(initial_images, station.joint_names)
            for name in ("policy_action", "leader_state"):
                features[f"intervention.{name}"] = {"dtype": "float32", "shape": (len(station.motor_ids),), "names": station.joint_names}
            for name in ("owner", "policy_valid", "leader_valid"):
                features[f"intervention.{name}"] = {"dtype": "int64", "shape": (1,), "names": [name]}
            root = args.dataset_root or (Path.home() / "datasets/rollout-interventions" / f"session-{time.time_ns()}")
            provenance = dict(collection, action="absolute follower goal after rollout protection",
                rollout_schema=1, ownership={"policy": 0, "human": 1, "transition": 2})
            recorder = RolloutRecorder(root, args.repo_id, fps,
                station.robot_type, features, provenance, args.record_queue_frames, wait=False)
            await asyncio.to_thread(recorder.wait_ready)
            print(f"Recording dataset: {recorder.root}")
        print(f"Checkpoint: {metadata['checkpoint']}; {fps} Hz; replan every {args.chunk_steps} steps")
        print("Space starts/pauses; q quits; s/f end episode as success/failure.")
        if args.teleop_takeover:
            print("t takes over; r resumes policy; grabbing the leader also takes over." if args.auto_takeover else "t takes over; r resumes policy (automatic grab detection disabled).")
        print("Support the arms when torque is released." if args.execute else "PREVIEW: no motor writes or training data recorded.")
        trajectory = None
        started = None
        last_status = 0.0
        last_request = float("-inf")
        last_control = None
        control_ticks = 0
        control_started = None
        inference_ms = roundtrip_ms = None
        handoff = None
        last_action = None
        retained_goals = None
        log_path = args.diagnostics_jsonl or (Path.home() / "benchmarks/white-lid-smolvla-20261017/rollouts" / f"{metadata.get('policy_type', 'policy')}-{time.time_ns()}.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = DiagnosticWriter(log_path)
        diagnostics.write(json.dumps({"event": "start", "checkpoint": metadata["checkpoint"], "execute": args.execute,
            "fps": fps, "backend": metadata.get("backend", "eager"), "replan_steps": args.chunk_steps,
            "deadband_ticks": args.deadband_ticks, "noise_seed": args.noise_seed,
            "blend_seconds": args.blend_seconds, "handoff_seconds": args.handoff_seconds,
            "takeover": args.teleop_takeover, "auto_takeover": args.auto_takeover,
            "leader_torque_limit": args.leader_torque_limit, "grab_error_ticks": args.grab_error_ticks,
            "grab_current": args.grab_current, "grab_seconds": args.grab_seconds,
            "dataset": str(recorder.root) if recorder else None}) + "\n")
        print(f"Diagnostics: {log_path}")

        def cancel_prediction():
            nonlocal pending, trajectory
            if pending is not None:
                pending.cancel()
                pending = None
            trajectory = None

        async def stop_episode(reason, outcome="unreviewed"):
            nonlocal armed, running, episode_open
            cancel_prediction()
            running = False
            try:
                if armed:
                    await station.disable(follower)
                    armed = False
            finally:
                if mirror and args.execute:
                    await mirror.stop()
            if recorder and episode_open:
                recorder.end(reason, outcome)
                episode_open = False
            diagnostics.status(f"Stopped episode: {reason}; outcome={outcome}. Space starts another trial.")

        deadline = time.monotonic()
        with keyboard() as keys:
            while True:
                if recorder:
                    recorder.check()
                key = keys.get_nowait() if not keys.empty() else None
                if key in ("q", "\x04"):
                    await stop_episode("quit")
                    break
                if running and (key in ("s", "f") or key == " " or time.monotonic() - started >= args.seconds):
                    reason = {"s": "success", "f": "failure", " ": "paused"}.get(key, "duration")
                    await stop_episode(reason, reason if reason in ("success", "failure") else "unreviewed")
                    continue
                if key == " " and not running:
                    source, motors, images = observe()
                    cancel_prediction()
                    retained_goals = [m.target_position for m in motors]
                    started = time.monotonic()
                    mode, running, handoff, last_action = "policy", True, None, None
                    last_request, last_control, control_started, control_ticks = float("-inf"), None, None, 0
                    station.protection = ProtectionTable()
                    if recorder:
                        recorder.begin({"checkpoint": metadata["checkpoint"], "backend": metadata.get("backend", "eager"),
                            "started_unix_ns": time.time_ns(), "task": collection["task"],
                            "deadband_ticks": args.deadband_ticks, "max_delta_ticks": args.max_delta_ticks,
                            "noise_seed": args.noise_seed, "teleop_takeover": args.teleop_takeover,
                            "auto_takeover": args.auto_takeover, "leader_torque_limit": args.leader_torque_limit,
                            "grab_error_ticks": args.grab_error_ticks, "grab_current": args.grab_current,
                            "grab_seconds": args.grab_seconds, "handoff_seconds": args.handoff_seconds,
                            "blend_seconds": args.blend_seconds, "replan_steps": args.chunk_steps})
                        episode_open = True
                    diagnostics.status("WAITING FOR FRESH PREDICTION (torque not enabled yet)")
                    deadline = time.monotonic()
                if not running:
                    await asyncio.sleep(.025)
                    continue
                observe_started = time.monotonic()
                source, motors, images = observe()
                sample_unix_ns = time.time_ns()
                motor_timestamps = {serial: [m.get_system_stamp_ns() for m in find_bus(station.motors.latest, serial).get_motors()]
                    for serial in ((leader, follower) if source else (follower,))}
                camera_timestamps = {name: station.cameras[camera].latest[1] for name, camera in cameras.items()}
                observe_ms = 1000 * (time.monotonic() - observe_started)
                now = time.monotonic()
                state = vector([m.present_position for m in motors], motors)
                leader_state = vector([m.present_position for m in source], source) if source else np.zeros_like(state)
                leader_reference = mirror_targets(source, motors, station.config) if source else None
                auto_grab = False
                if mirror and mode == "policy" and trajectory is not None and args.auto_takeover:
                    bus = find_bus(station.motors.latest, leader)
                    stamp = tuple(m.get_system_stamp_ns() for m in bus.get_motors())
                    auto_grab = detector.update(now, stamp, source, targets=leader_reference)
                if mirror and mode != "human" and (key == "t" or auto_grab) and (armed or not args.execute):
                    if args.execute:
                        await mirror.stop()
                    cancel_prediction()
                    mode = "human"
                    # Remove the initial mismatch, then decay that offset gradually.
                    origin = list(last_action if last_action is not None else state)
                    handoff = (now, origin, list(leader_state))
                    diagnostics.status("HUMAN TAKEOVER (grab)" if auto_grab else "HUMAN TAKEOVER (key)")
                if mirror and mode == "human" and key == "r":
                    mode, last_request = "resuming", float("-inf")
                    diagnostics.status("Requesting fresh policy prediction; human remains in control")
                if mode != "human" and pending is not None and pending.done():
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
                        "client_timings_ms": response.get("client_timings_ms"), "server_timings_ms": response.get("timings_ms")}) + "\n")
                    if not fresh:
                        last_request = float("-inf")
                        diagnostics.status("Discarded late prediction; requesting fresh observation")
                    else:
                        candidate = TimedTrajectory(observed_at, fps, chunk)
                        candidate.action_at(now)
                        if trajectory is not None:
                            candidate.splice_from(trajectory, now, args.blend_seconds)
                        else:
                            origin = list(last_action if last_action is not None else state)
                            handoff = (now, origin, candidate.action_at(now))
                            if recorder:
                                recorder.capacity()
                            if args.execute:
                                if not armed:
                                    armed = True
                                    retained_goals = [m.present_position for m in motors]
                                    # Seed follower goal before torque enable.
                                    from mirror import MotorCommand
                                    await send_motor_commands(station.client, follower,
                                        [MotorCommand(mid, speed=300, accel=5, goal=m.present_position)
                                         for mid, m in zip(station.motor_ids, motors, strict=True)])
                                    await set_torque(station.client, follower, station.motor_ids, enable=True)
                                if mirror:
                                    await mirror.start(source)
                            detector.reset(now)
                            mode = "policy"
                            diagnostics.status("POLICY CONTROL")
                        trajectory = candidate
                if mode != "human" and pending is None and now - last_request >= args.chunk_steps / fps:
                    last_request = now
                    pending = asyncio.create_task(predict_async(args.policy_url, now, images, state,
                        None if args.noise_seed == -1 else args.noise_seed))
                if trajectory is None and mode == "policy":
                    await asyncio.sleep(.005)
                    deadline = time.monotonic()
                    continue
                policy_action = trajectory.action_at(now) if trajectory is not None else None
                action = list(policy_action if mode == "policy" else leader_state)
                transitioning = handoff is not None and now - handoff[0] < args.handoff_seconds
                if transitioning:
                    fraction = max(0., 1 - (now - handoff[0]) / args.handoff_seconds)
                    action = [a + fraction * (origin - initial) for a, origin, initial in zip(action, handoff[1], handoff[2], strict=True)]
                action = [float(value) for value in action]
                owner = "transition" if transitioning or mode == "resuming" else mode
                commands = protected_commands(action, motors, station.protection, station.config, args.max_delta_ticks)
                if recorder:
                    recorder.capacity()
                send_started = time.monotonic()
                if args.execute:
                    await send_motor_commands(station.client, follower, commands)
                    if mirror and mirror.active:
                        await mirror.step(source, motors)
                send_ms = 1000 * (time.monotonic() - send_started)
                goals = list(retained_goals)
                for command in commands:
                    if command.goal is not None:
                        goals[command.motor_id - 1] = command.goal
                retained_goals = goals
                sent_state = vector(goals, motors)
                last_action = sent_state.tolist()
                detail = {"event": "control", "time_s": now - started, "capture_unix_ns": sample_unix_ns, "command_sent_unix_ns": time.time_ns(),
                    "owner": owner, "mode": mode, "interval_ms": None if last_control is None else 1000 * (now - last_control),
                    "action_age_ms": None if trajectory is None else 1000 * (now - trajectory.observed_at),
                    "predicted_action": policy_action, "requested_action": action,
                    "measured_state": state.tolist(), "protected_goal": sent_state.tolist(),
                    "currents": [m.current for m in motors], "leader_state": leader_state.tolist() if source else None,
                    "leader_currents": [m.current for m in source] if source else None,
                    "leader_targets": [m.target_position for m in source] if source else None,
                    "leader_tracking_targets": leader_reference,
                    "motor_source_timestamps_ns": motor_timestamps,
                    "camera_source_timestamps_ns": camera_timestamps,
                    "observe_ms": observe_ms, "send_ms": send_ms,
                    "commands": [{"motor_id": c.motor_id, "goal": c.goal, "speed": c.speed, "accel": c.accel} for c in commands]}
                diagnostics.write(json.dumps(detail) + "\n")
                if recorder:
                    frame = {"observation.state": state, "action": sent_state, "task": collection["task"], **images,
                        "intervention.policy_action": np.asarray(policy_action if policy_action is not None else np.zeros_like(state), dtype=np.float32),
                        "intervention.leader_state": leader_state,
                        "intervention.owner": np.asarray([{"policy": 0, "human": 1, "transition": 2}[owner]], dtype=np.int64),
                        "intervention.policy_valid": np.asarray([policy_action is not None], dtype=np.int64),
                        "intervention.leader_valid": np.asarray([source is not None], dtype=np.int64)}
                    recorder.add(frame, detail)
                if control_started is None:
                    control_started = now
                control_ticks += 1
                last_control = now
                if now - last_status >= 1:
                    hz = (control_ticks - 1) / max(now - control_started, 1e-6)
                    diagnostics.status(f"{now - started:4.1f}s {hz:4.1f}Hz {owner} infer={inference_ms or 0:.0f}ms rtt={roundtrip_ms or 0:.0f}ms | recording={'yes' if recorder else 'no'}")
                    last_status = now
                deadline += 1 / fps
                if deadline < time.monotonic():
                    deadline = time.monotonic() + 1 / fps
                await asyncio.sleep(max(0, deadline - time.monotonic()))
    finally:
        if pending is not None:
            pending.cancel()
        async def cleanup():
            try:
                if armed:
                    await station.disable(follower)
            finally:
                try:
                    if mirror and args.execute:
                        await mirror.restore()
                finally:
                    try:
                        if recorder is not None:
                            await asyncio.to_thread(recorder.close)
                    finally:
                        try:
                            if diagnostics is not None:
                                await asyncio.to_thread(diagnostics.close)
                        finally:
                            station.client.close_connection()
        task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


def parse_args():
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
    parser.add_argument("--teleop-takeover", action="store_true", help="Mirror follower onto leader and enable t/r handoffs")
    parser.add_argument("--auto-takeover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leader-torque-limit", type=int, default=100, help="Temporary RAM torque cap, raw units; never raises the original cap")
    parser.add_argument("--grab-error-ticks", type=int, default=80)
    parser.add_argument("--grab-current", type=int, default=40, help="Raw current threshold; tune on hardware")
    parser.add_argument("--grab-seconds", type=float, default=.15)
    parser.add_argument("--handoff-seconds", type=float, default=.3)
    parser.add_argument("--dataset-root", type=Path, help="Append-compatible rollout dataset; default: new local session directory")
    parser.add_argument("--repo-id", default="local/rollout-interventions")
    parser.add_argument("--no-record", action="store_true", help="Explicitly disable executed rollout recording")
    parser.add_argument("--record-queue-frames", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.leader_torque_limit <= 300 or args.grab_error_ticks <= 0 or args.grab_current <= 0:
        parser.error("Use torque limit 1–300 and positive grab thresholds")
    if any(not math.isfinite(v) or v <= 0 for v in (args.grab_seconds, args.handoff_seconds)) or args.record_queue_frames < 1:
        parser.error("Use finite positive grab/handoff durations and recording queue size")
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.chunk_steps <= 20 or not 0 < args.max_age <= 1 or args.max_delta_ticks <= 0:
        parser.error("Use positive duration/delta, 1–20 chunk steps, and max-age in (0, 1]")
    if not math.isfinite(args.blend_seconds) or not 0 <= args.blend_seconds <= 0.5:
        parser.error("--blend-seconds must be in [0, 0.5]")
    if not -1 <= args.noise_seed < 2**63:
        parser.error("--noise-seed must be -1 or an integer in [0, 2**63)")
    if not 1 <= args.deadband_ticks <= 20:
        parser.error("--deadband-ticks must be in [1, 20]")
    return args


def main():
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, ValueError, urllib.error.URLError) as exc:
        raise SystemExit(f"Stopped: {exc}")


if __name__ == "__main__":
    main()
