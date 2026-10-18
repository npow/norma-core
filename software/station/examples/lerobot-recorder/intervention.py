"""Hardware-independent takeover detection and bounded rollout recording."""
import hashlib
import json
import queue
import threading
import time
from pathlib import Path


class GrabDetector:
    """Require error AND effort on the same joint over distinct telemetry samples."""
    def __init__(self, error_ticks=80, current=40, seconds=.15, grace=.75):
        self.error_ticks, self.current = error_ticks, current
        self.seconds, self.grace = seconds, grace
        self.reset(0)

    def reset(self, now):
        self.enabled_at = now
        self.since = {}
        self.last_stamp = None

    def update(self, now, stamp, motors, max_steps=4096, targets=None):
        if stamp == self.last_stamp:
            return False
        self.last_stamp = stamp
        if now - self.enabled_at < self.grace:
            self.since.clear()
            return False
        triggered = False
        for i, motor in enumerate(motors):
            target = motor.target_position if targets is None else targets[i]
            distance = abs(motor.present_position - target) % max_steps
            distance = min(distance, max_steps - distance)
            if distance >= self.error_ticks and motor.current >= self.current:
                self.since.setdefault(i, now)
                triggered |= now - self.since[i] >= self.seconds
            else:
                self.since.pop(i, None)
        return triggered


def mirror_targets(source, follower, config):
    """Full follower pose in leader encoder coordinates, before per-tick slew limits."""
    from mirror import normalize_position, get_steps_range
    targets = []
    for leader_motor, follower_motor in zip(source, follower, strict=True):
        percent = normalize_position(follower_motor.present_position, follower_motor.range_min, follower_motor.range_max, config)
        width = get_steps_range(leader_motor.range_min, leader_motor.range_max, config)
        offset = min(width - config.safety_margin, max(config.safety_margin, percent / 100 * width))
        targets.append((leader_motor.range_min + round(offset)) % config.max_steps)
    return targets


class RolloutRecorder:
    """A single owner for dataset mutations; refuse unbounded capture backlogs."""
    def __init__(self, root, repo_id, fps, robot_type, features, provenance, max_queue=64, wait=True):
        self.root = Path(root).expanduser().resolve()
        self.queue = queue.Queue(maxsize=max_queue)
        self.error = None
        self.ready = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self._run,
            args=(repo_id, fps, robot_type, features, provenance, max_queue), daemon=True)
        self.thread.start()
        if wait:
            self.wait_ready()

    def wait_ready(self):
        self.ready.wait()
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Rollout recording failed: {self.error}") from self.error

    def capacity(self):
        self.check()
        if self.queue.full():
            raise RuntimeError("Rollout recording backlog full; stopping motion to preserve data")

    def put(self, kind, value=None):
        self.check()
        if self.closed:
            raise RuntimeError("Rollout recorder is closed")
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full:
            raise RuntimeError("Rollout recording backlog full; stopping motion to preserve data") from None

    def begin(self, metadata):
        self.put("begin", metadata)

    def add(self, frame, details):
        # Own the arrays before returning to the controller.
        self.put("frame", ({k: v.copy() if hasattr(v, "copy") else v for k, v in frame.items()}, details))

    def end(self, reason, outcome="unreviewed"):
        self.put("end", {"end_reason": reason, "outcome": outcome})

    def close(self):
        if self.closed:
            self.check()
            return
        # Motion must already be disabled. Drain even when the bounded queue is full.
        while self.thread.is_alive():
            try:
                self.queue.put(("close", None), timeout=.1)
                break
            except queue.Full:
                pass
        self.closed = True
        self.thread.join()
        self.check()

    def _run(self, *args):
        try:
            self._work(*args)
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()

    def _work(self, repo_id, fps, robot_type, features, provenance, max_queue):
        from record import dataset_lock, open_dataset, separate_episode_files, EpisodeCapture, quiet_video_logs
        dataset = capture = journal = None
        metadata = rows = sample_times = None
        digest = None

        def finish(ending):
            nonlocal capture, journal
            if capture is None:
                return
            current, capture = capture, None
            if journal is not None:
                journal.close()
                journal = None
            current.finish()
            count = current.buffer["size"]
            if count:
                dataset.save_episode(episode_data=current.buffer, parallel_encoding=False)
                dataset.finalize()
                latest = {key: value[0] for key, value in dataset.meta.latest_episode.items()}
                paths = [Path(dataset.meta.data_path.format(
                    chunk_index=latest["data/chunk_index"], file_index=latest["data/file_index"]))]
                paths += [Path(dataset.meta.video_path.format(video_key=key,
                    chunk_index=latest[f"videos/{key}/chunk_index"], file_index=latest[f"videos/{key}/file_index"]))
                    for key in dataset.meta.video_keys]
                paths += [Path("meta/interventions") / f"episode-{current.index:06d}.jsonl"]
                file_hashes = {}
                for relative in paths:
                    with (self.root / relative).open("rb") as stream:
                        file_hashes[str(relative)] = hashlib.file_digest(stream, "sha256").hexdigest()
                audit = dict(metadata, **ending, frames=count, frame_sha256=digest.hexdigest(), file_sha256=file_hashes,
                    capture_time_ns=sample_times, storage_time_ns=current.capture_times_ns, control_counts=rows,
                    incomplete=ending["end_reason"] not in ("success", "failure", "duration", "paused"))
                path = self.root / "meta/interventions" / f"episode-{current.index:06d}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(audit, indent=2) + "\n")
                temporary.replace(path)
                print(f"Saved rollout {current.index:06d}: {count} frames; ownership={rows}; outcome={ending['outcome']}", flush=True)
            else:
                current.discard()
            capture = None

        try:
            with dataset_lock(self.root):
                try:
                    quiet_video_logs()
                    dataset = open_dataset(self.root, repo_id, fps, robot_type, features, provenance)
                    separate_episode_files(dataset)
                    directory = self.root / "meta/interventions"
                    for index in range(dataset.meta.total_episodes):
                        if not (directory / f"episode-{index:06d}.json").is_file() or not (directory / f"episode-{index:06d}.jsonl").is_file():
                            raise RuntimeError(f"Episode {index}: missing intervention audit; preserve this dataset and use a new root")
                    # An abrupt exit may leave the useful joint/action journal without a finalized episode.
                    recovery = self.root.parent / f"{self.root.name}-recovery" / f"journals-{time.time_ns()}"
                    for path in directory.glob("episode-*"):
                        index = int(path.name.split(".")[0].removeprefix("episode-"))
                        if index >= dataset.meta.total_episodes:
                            recovery.mkdir(parents=True, exist_ok=True)
                            path.rename(recovery / path.name)
                    self.ready.set()
                    while True:
                        kind, value = self.queue.get()
                        if kind == "begin":
                            if capture is not None:
                                raise RuntimeError("Episode already active")
                            capture = EpisodeCapture(dataset, dataset.meta.total_episodes)
                            sample_times = []
                            metadata, rows, digest = value, {"policy": 0, "human": 0, "transition": 0}, hashlib.sha256()
                            directory = self.root / "meta/interventions"
                            directory.mkdir(parents=True, exist_ok=True)
                            journal = (directory / f"episode-{capture.index:06d}.jsonl").open("w", buffering=1)
                        elif kind == "frame":
                            if capture is None:
                                raise RuntimeError("No active rollout episode")
                            if sum(not f.done() for f in capture.pending) >= max_queue * max(1, len(capture.paths)):
                                raise RuntimeError("Image writer backlog full")
                            frame, details = value
                            capture.add(frame)
                            for key in sorted(frame):
                                item = frame[key]
                                digest.update(key.encode())
                                digest.update(item.tobytes() if hasattr(item, "tobytes") else str(item).encode())
                            sample_times.append(details.get("capture_unix_ns", capture.capture_times_ns[-1]))
                            rows[details["owner"]] += 1
                            journal.write(json.dumps(dict(details, frame_index=capture.buffer["size"] - 1)) + "\n")
                        elif kind == "end":
                            finish(value)
                        elif kind == "close":
                            finish({"end_reason": "interrupted", "outcome": "unreviewed"})
                            return
                finally:
                    try:
                        finish({"end_reason": "recording_error", "outcome": "unreviewed"})
                    finally:
                        if dataset is not None:
                            dataset.finalize()
                            dataset.stop_image_writer()
        except BaseException as exc:
            self.error = exc
        finally:
            if journal is not None:
                journal.close()
            self.ready.set()


class LeaderMirror:
    """Temporary RAM settings only; keep separate protection from the follower."""
    def __init__(self, station, serial, torque_limit, max_delta):
        from mirror import ProtectionTable
        from state import find_bus
        self.station, self.serial = station, serial
        self.torque_limit, self.max_delta = torque_limit, max_delta
        self.protection = ProtectionTable()
        self.active = False
        self.touched = False
        bus = find_bus(station.motors.latest, serial)
        self.original = [(m.get_id(), bytes(m.get_state())[0x30:0x32]) for m in bus.get_motors()]
        if any(len(value) != 2 for _, value in self.original):
            raise ValueError("Missing leader torque limit telemetry")

    async def start(self, motors):
        from commands import _sync_write, set_torque, send_motor_commands
        from mirror import MotorCommand, ProtectionTable
        from software.station.shared.station_py import send_commands
        if any(m.error_status for m in motors):
            raise RuntimeError("Leader fault before enabling mirroring")
        # Mark before writing so even partial failures restore the settings.
        self.touched = True
        await set_torque(self.station.client, self.serial, self.station.motor_ids, False)
        await send_commands(self.station.client, [_sync_write(self.serial, 0x30,
            [(mid, min(self.torque_limit, int.from_bytes(original, "little")).to_bytes(2, "little"))
             for mid, original in self.original])])
        # Clear stale goals before enabling. Never snap to the follower on enable.
        await send_motor_commands(self.station.client, self.serial,
            [MotorCommand(mid, speed=300, accel=5, goal=m.present_position)
             for mid, m in zip(self.station.motor_ids, motors, strict=True)])
        await set_torque(self.station.client, self.serial, self.station.motor_ids, True)
        self.protection = ProtectionTable()
        self.active = True

    async def stop(self):
        from commands import set_torque
        if self.touched:
            await set_torque(self.station.client, self.serial, self.station.motor_ids, False)
        self.active = False

    async def restore(self):
        from commands import _sync_write
        from software.station.shared.station_py import send_commands
        if self.touched:
            # Never raise the torque cap if disabling torque failed.
            await self.stop()
            await send_commands(self.station.client, [_sync_write(self.serial, 0x30, self.original)])

    async def step(self, leader_motors, follower_motors):
        from commands import send_motor_commands
        from station_io import vector
        from rollout_act import protected_commands
        if any(m.error_status for m in leader_motors):
            raise RuntimeError("Leader fault while actively mirroring")
        commands = protected_commands(vector([m.present_position for m in follower_motors], follower_motors),
            leader_motors, self.protection, self.station.config, self.max_delta)
        await send_motor_commands(self.station.client, self.serial, commands)
