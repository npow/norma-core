#!/usr/bin/env python3
"""Space: start/stop and save. d: delete current/last take. q: quit."""
import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import math
import re
import sys
import shutil
import tempfile
import termios
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


async def blocking(function, *args):
    """Finish a disk mutation before allowing Ctrl-C to run finalization."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def separate_episode_files(dataset):
    # Keep software video encoding from taking all cores during live capture.
    dataset._encoder_threads = 1
    # LeRobot rotates files between whole episodes, never inside one. A
    # one-byte target gives each episode its own ascending file number.
    dataset.meta.update_chunk_settings(
        data_files_size_in_mb=1 / 1024**2, video_files_size_in_mb=1 / 1024**2,
    )


def quiet_video_logs():
    import av
    from datasets.utils.tqdm import disable_progress_bars
    # LeRobot restores FFmpeg's native callback between videos. Set the C
    # library level too, so x264 statistics stay hidden after that reset.
    av.logging.set_libav_level(av.logging.ERROR)
    av.logging.set_level(av.logging.ERROR)
    logging.getLogger("libav").setLevel(logging.ERROR)
    disable_progress_bars()


@contextlib.contextmanager
def dataset_lock(root):
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.recorder.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another recorder is already using {root}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def preserve_unfinished_images(root):
    """Move interrupted take images aside without deleting or copying their contents."""
    images = root / "images"
    if not any(path.is_file() for path in images.rglob("*")):
        return None
    recovery = root.parent / f"{root.name}-recovery" / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    recovery.mkdir(parents=True)
    images.rename(recovery / "images")
    print(f"Preserved interrupted take images at {recovery}. Continuing from the last saved episode.", flush=True)
    return recovery


def open_dataset(root, repo_id, fps, robot_type, features, provenance, remap_cameras=False):
    """Create a dataset or validate and reopen it for appending, entirely locally."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if not root.exists():
        dataset = LeRobotDataset.create(
            repo_id=repo_id, root=root, fps=fps, robot_type=robot_type,
            features=features, use_videos=True, vcodec="h264", image_writer_threads=2,
        )
        (root / "meta/collection.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return dataset

    info_path, collection_path = root / "meta/info.json", root / "meta/collection.json"
    if not info_path.is_file() or not collection_path.is_file():
        raise ValueError(f"Cannot resume {root}: expected a dataset created by this recorder")
    info = json.loads(info_path.read_text())
    previous = json.loads(collection_path.read_text())
    cameras_changed = previous.get("cameras") != provenance.get("cameras")
    if remap_cameras and set(previous.get("cameras", {})) != set(provenance.get("cameras", {})):
        raise ValueError("Cannot remap: use exactly the saved camera names")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Cannot resume: expected LeRobot v3.0")
    if info.get("fps") != fps or info.get("robot_type") != robot_type:
        raise ValueError("Cannot resume: FPS or robot type differs from the saved dataset")
    for key in ("leader_bus", "follower_bus", "cameras", "calibration_ranges", "motor_ids", "units", "action"):
        if key == "cameras" and remap_cameras:
            continue
        if previous.get(key) != provenance.get(key):
            if key == "cameras":
                raise ValueError("Cannot resume: cameras differs from the saved collection; "
                                 "after USB re-enumeration, use --remap-cameras with "
                                 "--camera NAME=QUEUE for every saved view (keep physical views matched)")
            raise ValueError(f"Cannot resume: {key} differs from the saved collection")
    standard = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if set(info["features"]) - standard != set(features):
        raise ValueError("Cannot resume: camera or joint features differ")
    for key, feature in features.items():
        saved = info["features"][key]
        if any(saved.get(field) != (list(feature[field]) if field == "shape" else feature.get(field))
               for field in ("dtype", "shape", "names")):
            raise ValueError(f"Cannot resume: shape, names or dtype changed for {key}")

    total_episodes = info.get("total_episodes", 0)
    meta = None
    if total_episodes:
        if not (root / "meta/tasks.parquet").is_file() or not (root / "meta/stats.json").is_file() or not list((root / "meta/episodes").rglob("*.parquet")):
            raise ValueError("Cannot resume: episode metadata is incomplete")
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    elif info.get("total_frames") != 0:
        raise ValueError("Cannot resume: empty dataset has a nonzero frame count")
    referenced_data, referenced_videos = set(), set()
    if total_episodes:
        if len(meta.episodes) != meta.total_episodes:
            raise ValueError("Cannot resume: episode metadata is incomplete")
        for index, episode in enumerate(meta.episodes):
            if episode["episode_index"] != index:
                raise ValueError("Cannot resume: episode indexes are not contiguous")
            referenced_data.add(root / meta.get_data_file_path(index))
            referenced_videos.update(root / meta.get_video_file_path(index, key) for key in meta.video_keys)
    if referenced_data != set((root / "data").rglob("*.parquet")) or referenced_videos != set((root / "videos").rglob("*.mp4")):
        raise ValueError("Cannot resume: missing or unindexed episode files; existing data was preserved")
    if total_episodes:
        dataset = LeRobotDataset(repo_id, root=root, vcodec="h264")
        if len(dataset.hf_dataset) != meta.total_frames:
            raise ValueError("Cannot resume: frame count differs from metadata")
    else:
        # The normal LeRobot loader expects a data parquet file even for an
        # empty dataset. Build its empty buffers separately, preserving root.
        with tempfile.TemporaryDirectory(prefix=".recorder-empty-", dir=root.parent) as temp:
            dataset = LeRobotDataset.create(
                repo_id=repo_id, root=Path(temp) / "empty", fps=fps,
                robot_type=robot_type, features=features, use_videos=True, vcodec="h264",
            )
            dataset.root = root
            dataset.meta.root = root
    preserve_unfinished_images(root)
    dataset.start_image_writer(num_threads=2)
    if cameras_changed and remap_cameras:
        previous.setdefault("camera_remaps", []).append({
            "from_episode": total_episodes, "previous": previous["cameras"],
            "current": provenance["cameras"], "timestamp_unix": time.time(),
        })
        previous["cameras"] = provenance["cameras"]
        pending = collection_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(previous, indent=2) + "\n")
        pending.replace(collection_path)
        print(f"Camera mapping updated from episode {total_episodes:06d}; previous mapping retained in meta/collection.json.", flush=True)
    print(f"Resuming {total_episodes} saved episodes; next episode {total_episodes:06d}.", flush=True)
    return dataset


class EpisodeCapture:
    """Own one take's buffers and PNG writes independently of the dataset writer."""
    def __init__(self, dataset, index):
        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
        self.root = dataset.root
        self.features = dataset.features
        self.fps = dataset.fps
        self.buffer = dataset.create_episode_buffer(episode_index=index)
        self.index = index
        self.paths = {key: self.root / DEFAULT_IMAGE_PATH.format(
            image_key=key, episode_index=index, frame_index=0,
        ) for key, feature in self.features.items() if feature["dtype"] in ("video", "image")}
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"images-{index}")
        self.pending = []
        self.capture_times_ns = []

    @staticmethod
    def write_image(image, path):
        from PIL import Image
        Image.fromarray(image).save(path, compress_level=1)

    def add(self, frame):
        from lerobot.datasets.utils import validate_frame
        validate_frame(frame, self.features)
        unfinished = []
        for future in self.pending:
            if future.done():
                future.result()  # Surface disk/image errors during capture.
            else:
                unfinished.append(future)
        self.pending = unfinished
        index = self.buffer["size"]
        self.capture_times_ns.append(time.time_ns())
        self.buffer["frame_index"].append(index)
        self.buffer["timestamp"].append(frame.get("timestamp", index / self.fps))
        self.buffer["task"].append(frame["task"])
        for key, value in frame.items():
            if key in ("task", "timestamp"):
                continue
            if key in self.paths:
                path = self.paths[key].with_name(f"frame-{index:06d}.png")
                self.pending.append(self.writer.submit(self.write_image, value.copy(), path))
                self.buffer[key].append(str(path))
            else:
                self.buffer[key].append(value.copy())
        self.buffer["size"] += 1

    def finish(self):
        self.writer.shutdown(wait=True)
        for future in self.pending:
            future.result()
        self.pending.clear()

    def discard(self):
        self.finish()
        for path in self.paths.values():
            if path.parent.exists():
                shutil.rmtree(path.parent)


class Takes:
    """Save on stop, stay ready, and let d undo the most recent episode."""
    def __init__(self, dataset, max_frames=None, capture_factory=EpisodeCapture):
        quiet_video_logs()
        self.dataset = dataset
        separate_episode_files(dataset)
        self.recording = False
        self.frames = 0
        self.saved = dataset.meta.total_episodes
        self.next_episode = self.saved
        self.capture_factory = capture_factory
        self.capture = None
        self.saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-save")
        self.pending = []
        self.save_error = None
        self.started = 0.0
        self.last_status = 0.0
        self.max_frames = max_frames

    def ready(self):
        print(f"READY — episode {self.next_episode:06d} | {sum(not f.done() for f in self.pending)} saves pending | Space starts | d deletes last | q quits", flush=True)

    def check_saves(self):
        for future in self.pending:
            if future.done():
                future.result()
        self.pending = [future for future in self.pending if not future.done()]

    def flush(self):
        for future in self.pending:
            future.result()
        self.pending.clear()

    def save_capture(self, capture, count):
        try:
            capture.finish()
            if self.save_error:
                raise RuntimeError("An earlier background save failed") from self.save_error
            # Only this worker touches LeRobot metadata/parquet/video writers.
            # Explicit episode_data keeps capture buffers independent.
            self.dataset.save_episode(episode_data=capture.buffer, parallel_encoding=False)
            self.dataset.finalize()
            if hasattr(capture, "capture_times_ns"):
                timing = self.dataset.root / "meta/recording" / f"episode-{capture.index:06d}.json"
                timing.parent.mkdir(parents=True, exist_ok=True)
                timing.write_text(json.dumps({"capture_time_ns": capture.capture_times_ns}) + "\n")
            self.saved += 1
            print(f"\nSaved episode {capture.index:06d}: {count} frames", flush=True)
        except Exception as exc:
            self.save_error = exc
            raise RuntimeError(f"Background save failed for episode {capture.index:06d}: {exc}") from exc

    def commit(self):
        if self.frames:
            capture, count = self.capture, self.frames
            self.pending.append(self.saver.submit(self.save_capture, capture, count))
            print(f"\nQueued episode {self.next_episode:06d}: {count} frames; saving in background.", flush=True)
            self.next_episode += 1
            self.capture = None
            self.frames = 0
        elif self.capture:
            self.capture.discard()
            self.capture = None

    def space(self):
        self.check_saves()
        if self.recording:
            self.recording = False
            self.commit()
            self.ready()
        else:
            self.capture = self.capture_factory(self.dataset, self.next_episode)
            self.recording = True
            self.started = time.monotonic()
            self.last_status = self.started
            print(f"RECORDING episode {self.next_episode:06d} — Space stops + queues save; d discards", flush=True)

    def discard(self):
        if not self.recording and not self.frames and (self.saved or self.pending):
            self.flush()  # Deletion must not race an encoder or metadata update.
            self.delete_last()
            self.next_episode = self.saved
            self.ready()
            return
        count = self.frames
        if self.capture:
            self.capture.discard()
            self.capture = None
        self.frames = 0
        self.recording = False
        print(f"\nDiscarded episode {self.next_episode:06d}: {count} frames.", flush=True)
        self.ready()

    def delete_last(self):
        print(f"Deleting episode {self.saved - 1:06d}…", flush=True)
        from lerobot.datasets.dataset_tools import delete_episodes
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset.finalize()
        self.dataset.stop_image_writer()
        root, repo_id = self.dataset.root, self.dataset.repo_id
        source = LeRobotDataset(repo_id, root=root)
        with tempfile.TemporaryDirectory(prefix=".recorder-delete-", dir=root.parent) as temp:
            replacement = Path(temp) / "replacement"
            if self.saved > 1:
                filtered = delete_episodes(source, [self.saved - 1], output_dir=replacement, repo_id=repo_id)
                filtered.finalize()
            else:
                filtered = LeRobotDataset.create(
                    repo_id=repo_id, root=replacement, fps=source.fps,
                    robot_type=source.meta.robot_type, features=source.features,
                    use_videos=bool(source.meta.video_keys), vcodec="h264",
                )
            collection = root / "meta/collection.json"
            if collection.exists():
                shutil.copy2(collection, replacement / "meta/collection.json")
            for timing in (root / "meta/recording").glob("episode-*.json"):
                if int(timing.stem.removeprefix("episode-")) < self.saved - 1:
                    destination = replacement / "meta/recording" / timing.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(timing, destination)
            backup = Path(temp) / "original"
            root.rename(backup)
            try:
                replacement.rename(root)
                if self.saved > 1:
                    self.dataset = LeRobotDataset(repo_id, root=root)
                else:
                    filtered.root = root
                    filtered.meta.root = root
                    self.dataset = filtered
                separate_episode_files(self.dataset)
                self.dataset.vcodec = "h264"
                self.dataset.start_image_writer(num_threads=2)
            except Exception:
                if root.exists():
                    root.rename(Path(temp) / "failed")
                backup.rename(root)
                self.dataset = LeRobotDataset(repo_id, root=root)
                raise
        self.saved -= 1
        print(f"Deleted episode {self.saved:06d}.", flush=True)

    def add(self, frame):
        self.check_saves()
        self.capture.add(frame)
        self.frames += 1
        now = time.monotonic()
        if now - self.last_status >= 1:
            backlog = sum(not future.done() for future in getattr(self.capture, "pending", []))
            print(f"\rRECORDING episode {self.next_episode:06d} | {self.frames} frames | {now - self.started:.1f}s | {backlog} image writes pending", end="", flush=True)
            self.last_status = now
        if self.recording and self.max_frames is not None and self.frames >= self.max_frames:
            print("\nEpisode duration reached; stopping and saving.", flush=True)
            self.space()

    def finish(self):
        self.recording = False
        self.commit()
        if self.pending:
            print("Finishing background saves before exit…", flush=True)
        try:
            self.flush()
        finally:
            self.saver.shutdown(wait=True)
            self.dataset.finalize()
            self.dataset.stop_image_writer()


@contextlib.contextmanager
def keyboard():
    if not sys.stdin.isatty():
        raise RuntimeError("Recording requires an interactive terminal")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    keys = asyncio.Queue()
    loop = asyncio.get_running_loop()
    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, lambda: keys.put_nowait(sys.stdin.read(1)))
        yield keys
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def camera_mapping(specs, available):
    if not specs:
        return {f"camera_{i}": queue for i, queue in enumerate(sorted(available))}
    cameras = {}
    for spec in specs:
        name, separator, queue = spec.partition("=")
        if not separator or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name):
            raise ValueError("Use --camera NAME=usbvideo/QUEUE (letters, numbers, underscores)")
        if name in cameras or queue in cameras.values():
            raise ValueError("Camera names and queues must be unique")
        if queue not in available:
            raise ValueError(f"Camera queue {queue!r} not found; run --inspect")
        cameras[name] = queue
    return cameras


def features_for(images, joint_names):
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
        "action": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
    }
    for key, image in images.items():
        features[key] = {"dtype": "video", "shape": image.shape,
                         "names": ["height", "width", "channels"]}
    return features


async def run(args):
    from bootstrap import setup
    setup()
    from station_io import Station, vector
    # Import the ML/video stack before subscribing: importing it can take
    # several seconds and otherwise stalls the Station receive loop.
    if not args.inspect:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        quiet_video_logs()

    station = await Station.connect(args.server)
    await station.discover()
    station.describe()
    station.configure(args.leader_bus, args.follower_bus)
    cameras = camera_mapping(args.camera, station.cameras)
    _, _, images, calibration = station.snapshot(args.leader_bus, args.follower_bus, cameras, args.max_age)
    print(f"Ready: {station.robot_type}, {len(station.motor_ids)} motors, {len(cameras)} cameras; calibration and source ages checked.")
    if args.inspect:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Recording requires an interactive terminal")
    if args.leader_bus == args.follower_bus:
        raise ValueError("Leader and follower must be different buses")
    root = args.root.expanduser().resolve()
    expected_shapes = {key: image.shape for key, image in images.items()}
    provenance = {
        "server": args.server, "leader_bus": args.leader_bus, "follower_bus": args.follower_bus,
        "cameras": cameras, "calibration_ranges": calibration, "fps": args.fps,
        "task": args.task, "motor_ids": list(station.motor_ids),
        "units": "joints: calibrated [-100,100]; gripper: calibrated [0,100]",
        "action": "absolute follower goal after mirroring deadband and current protection",
        "synchronization": "latest source frames; bounded age/skew; constant-rate sampling",
        "max_source_age_and_skew_s": args.max_age,
    }
    dataset = await blocking(
        open_dataset, root, args.repo_id, args.fps, station.robot_type,
        features_for(images, station.joint_names), provenance, args.remap_cameras,
    )
    takes = Takes(dataset, math.ceil(args.episode_seconds * args.fps) if args.episode_seconds else None)
    armed = False
    period = 1 / args.fps
    deadline = time.monotonic()
    previous_error = None
    print(f"Output: {root}\nLeader {args.leader_bus} → follower {args.follower_bus}")
    print("Space starts teleop + recording. Between takes, teleop continues. q saves + quits.")
    print("Space stops and queues a background save. You can start the next episode immediately.")
    takes.ready()
    if args.episode_seconds:
        print(f"Each take stops automatically after {args.episode_seconds:g}s. Space can stop it earlier.")
    try:
        with keyboard() as keys:
            while True:
                takes.check_saves()
                key = keys.get_nowait() if not keys.empty() else None
                if key in ("q", "\x04"):
                    break
                if key == "d":
                    await blocking(takes.discard)
                if key == " " and takes.recording:
                    await blocking(takes.space)
                    deadline = time.monotonic()
                    continue
                try:
                    source, target, images, current_calibration = station.snapshot(
                        args.leader_bus, args.follower_bus, cameras, args.max_age,
                    )
                    if current_calibration != calibration:
                        raise RuntimeError("Calibration changed during collection; restart into a new dataset")
                    if any(image.shape != expected_shapes[k] for k, image in images.items()):
                        raise RuntimeError("Camera resolution changed during collection")
                    previous_error = None
                except RuntimeError as exc:
                    if takes.recording:
                        takes.recording = False
                        await blocking(takes.commit)
                        takes.ready()
                    if armed:
                        await station.disable(args.follower_bus)
                        armed = False
                    if str(exc) != previous_error:
                        print(f"Paused: {exc}. Fix the source, then Space to restart.", flush=True)
                        previous_error = str(exc)
                    await asyncio.sleep(period)
                    deadline = time.monotonic()
                    continue
                if key == " ":
                    if not takes.recording and not armed:
                        # Mark before enabling, so partial enable failure is cleaned up.
                        armed = True
                        await station.enable(args.leader_bus, args.follower_bus)
                    await blocking(takes.space)
                    deadline = time.monotonic()
                    # Read fresh observations after changing episode state.
                    await asyncio.sleep(period)
                    continue
                if armed:
                    action = await station.step(source, target, args.follower_bus)
                    if takes.recording:
                        frame = {"observation.state": vector([m.present_position for m in target], target),
                                 "action": action, "task": args.task, **images}
                        await blocking(takes.add, frame)
                deadline += period
                if deadline < time.monotonic() - period:
                    # A scheduling hiccup does not end the user's take or
                    # trigger catch-up bursts. Capture times are saved separately.
                    deadline = time.monotonic()
                await asyncio.sleep(max(0, deadline - time.monotonic()))
                if not takes.recording:
                    deadline = time.monotonic()
    finally:
        # Stop motion before encoding the last episode. Shield cleanup from Ctrl-C cancellation.
        async def cleanup():
            try:
                if armed:
                    await station.disable(args.follower_bus)
            finally:
                await asyncio.to_thread(takes.finish)
        task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
    print(f"Dataset ready: {root} ({takes.saved} episodes)")
    if takes.saved:
        print(f"View locally: uv run lerobot-dataset-viz --repo-id {args.repo_id} --root '{root}' --episode-index 0 --num-workers 0")
    if args.push_to_hub and takes.saved:
        await asyncio.to_thread(takes.dataset.push_to_hub, private=True)
        print(f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={args.repo_id}/episode_0")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--leader-bus", default="5B3D044117")
    parser.add_argument("--follower-bus", default="5B3D045222")
    parser.add_argument("--root", type=Path, default=Path("datasets") / time.strftime("demo-%Y%m%d-%H%M%S"))
    parser.add_argument("--repo-id", default="local/arm-demos")
    parser.add_argument("--task", default="", help="Natural-language instruction; required when recording")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=QUEUE")
    parser.add_argument("--remap-cameras", action="store_true",
                        help="Accept changed USB IDs; explicitly assign every saved view with --camera")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--episode-seconds", type=float, help="Automatically stop and save each take after this many seconds")
    parser.add_argument("--max-age", type=float, default=0.2, help="Maximum source age and inter-source skew in seconds")
    parser.add_argument("--inspect", action="store_true", help="List buses/cameras without moving or recording")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload privately after q (requires HF login)")
    args = parser.parse_args(argv)
    if args.remap_cameras and not args.camera:
        parser.error("--remap-cameras requires explicit --camera NAME=QUEUE assignments for every saved view")
    if not args.inspect and not args.task.strip():
        parser.error("--task is required for recording")
    if not 1 <= args.fps <= 60 or not 0 < args.max_age <= 1:
        parser.error("--fps must be 1–60; --max-age must be >0 and <=1 second")
    if args.episode_seconds is not None and (not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
        parser.error("--episode-seconds must be a finite positive duration")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", args.repo_id):
        parser.error("--repo-id must be OWNER/NAME")
    return args


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        args = parse_args()
        with contextlib.nullcontext() if args.inspect else dataset_lock(args.root):
            asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Stopped; cleanup completed.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
