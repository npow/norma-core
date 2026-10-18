"""Control transitions and a real two-camera LeRobot write/read round trip."""
import asyncio
import hashlib
import json
import tempfile
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np

from record import EpisodeCapture, Takes, blocking, camera_mapping, dataset_lock, features_for, open_dataset, parse_args


class MemoryCapture:
    def __init__(self, dataset, index):
        self.index = index
        self.buffer = {"episode_index": index, "frames": []}

    def add(self, frame):
        self.buffer["frames"].append(frame.copy())

    def finish(self):
        pass

    def discard(self):
        pass


class ControlsTest(unittest.TestCase):
    def test_recording_is_not_blocked_by_more_than_four_pending_saves(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        release = threading.Event()
        dataset.save_episode.side_effect = lambda **kwargs: release.wait(timeout=5)
        takes = Takes(dataset, capture_factory=MemoryCapture)
        try:
            for episode in range(6):
                takes.space()
                self.assertTrue(takes.recording)
                takes.add({"sample": episode})
                takes.space()
            takes.space()
            self.assertTrue(takes.recording)
            self.assertEqual(takes.next_episode, 6)
            release.set()
            takes.finish()
            self.assertEqual(takes.saved, 6)
        finally:
            release.set()
            takes.saver.shutdown(wait=True)

    def test_discard_current_does_not_commit(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        takes = Takes(dataset, capture_factory=MemoryCapture)
        takes.space()
        takes.add({"sample": 1})
        dataset.save_episode.assert_not_called()
        takes.discard()
        takes.finish()
        dataset.save_episode.assert_not_called()
        dataset.add_frame.assert_not_called()
        dataset.finalize.assert_called_once()

    def test_next_take_and_quit_commit_once(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        takes = Takes(dataset, capture_factory=MemoryCapture)
        takes.space()
        takes.add({"sample": 1})
        takes.space()
        takes.flush()
        self.assertEqual(dataset.save_episode.call_count, 1)
        self.assertFalse(takes.recording)
        self.assertEqual(takes.saved, 1)
        takes.space()
        self.assertEqual(dataset.save_episode.call_count, 1)
        takes.add({"sample": 2})
        takes.finish()
        self.assertEqual(dataset.save_episode.call_count, 2)

    def test_camera_order_and_ambiguity(self):
        self.assertEqual(camera_mapping([], ["b", "a"]), {"camera_0": "a", "camera_1": "b"})
        with self.assertRaises(ValueError):
            camera_mapping(["front=a", "wrist=a"], ["a"])
        with self.assertRaises(ValueError):
            camera_mapping(["front=missing"], ["a"])

    def test_duration_stops_saves_and_waits_for_next_space(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        takes = Takes(dataset, max_frames=3, capture_factory=MemoryCapture)
        takes.space()
        for _ in range(3):
            takes.add({"sample": 1})
        self.assertFalse(takes.recording)
        takes.flush()
        self.assertEqual(takes.saved, 1)
        self.assertEqual(takes.frames, 0)
        dataset.save_episode.assert_called_once_with(episode_data=ANY, parallel_encoding=False)
        takes.space()
        self.assertTrue(takes.recording)
        self.assertEqual(takes.saved, 1)
        takes.finish()

    def test_next_episode_records_while_save_is_blocked_and_quit_drains_queue(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        started, release = threading.Event(), threading.Event()

        def slow_save(**kwargs):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test save was never released")

        dataset.save_episode.side_effect = slow_save
        takes = Takes(dataset, capture_factory=MemoryCapture)
        try:
            takes.space()
            takes.add({"sample": 1})
            takes.space()
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(takes.saved, 0)
            takes.space()
            takes.add({"sample": 2})
            self.assertTrue(takes.recording)
            self.assertEqual(takes.frames, 1)
            self.assertEqual(takes.next_episode, 1)
            release.set()
            takes.finish()  # Also queues and saves the still-recording second take.
            self.assertEqual(takes.saved, 2)
            self.assertEqual([call.kwargs["episode_data"] for call in dataset.save_episode.call_args_list],
                             [{"episode_index": 0, "frames": [{"sample": 1}]},
                              {"episode_index": 1, "frames": [{"sample": 2}]}])
        finally:
            release.set()
            takes.saver.shutdown(wait=True)

    def test_background_save_error_is_reported(self):
        dataset = Mock()
        dataset.meta.total_episodes = 0
        dataset.save_episode.side_effect = OSError("disk full")
        takes = Takes(dataset, capture_factory=MemoryCapture)
        takes.space()
        takes.add({"sample": 1})
        takes.space()
        with self.assertRaisesRegex(RuntimeError, "Background save failed.*disk full"):
            takes.finish()

    def test_task_required(self):
        self.assertEqual(parse_args(["--inspect"]).leader_bus, "5B3D044117")
        with self.assertRaises(SystemExit):
            parse_args([])


class DatasetRoundTripTest(unittest.TestCase):
    def test_video_state_action_task_and_discard(self):
        from bootstrap import setup
        setup()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import av
        av.logging.set_level(av.logging.ERROR)
        from station_io import decode_camera, vector
        from target.gen_python.protobuf.drivers.usbvideo import frame as frame_pb, usbvideo

        # Verify Station's planar camera format, independently of the writer.
        image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        envelope = usbvideo.RxEnvelope(frames=frame_pb.FramesPack(
            format=frame_pb.FrameFormat(width=64, height=48, kind=0),
            stamps=[frame_pb.FrameStamp(local_stamp_ns=123)],
            linear_data=image.transpose(2, 0, 1).tobytes(),
        ))
        decoded, stamp, _ = decode_camera(envelope.encode())
        np.testing.assert_array_equal(decoded, image)
        self.assertEqual(stamp, 123)

        motors = [SimpleNamespace(range_min=100, range_max=1100)] * 8
        np.testing.assert_array_equal(vector([100] * 8, motors), [-100] * 7 + [0])
        np.testing.assert_array_equal(vector([1100] * 8, motors), [100] * 8)

        images = {f"observation.images.camera_{i}": np.full((48, 64, 3), i * 50, np.uint8) for i in range(2)}
        names = [f"motor_{i}.pos" for i in range(1, 9)]
        with tempfile.TemporaryDirectory(prefix="norma-recorder-test-") as temp:
            root = Path(temp) / "dataset"
            dataset = LeRobotDataset.create(
                repo_id="local/test", root=root, fps=20, robot_type="elrobot",
                features=features_for(images, names), use_videos=True, vcodec="h264",
                image_writer_threads=2,
            )
            takes = Takes(dataset)
            takes.space()
            takes.add({"observation.state": np.zeros(8, np.float32), "action": np.zeros(8, np.float32),
                       "task": "discard me", **images})
            takes.discard()
            expected = []
            expected_images = []
            saving, release_save = threading.Event(), threading.Event()
            original_save = dataset.save_episode

            def delayed_save(**kwargs):
                saving.set()
                if not release_save.wait(timeout=5):
                    raise RuntimeError("Timed out waiting for concurrent camera capture")
                return original_save(**kwargs)

            dataset.save_episode = delayed_save
            stop_times = []
            for episode in range(2):
                takes.space()
                for index in range(5):
                    state = np.full(8, episode * 10 + index, np.float32)
                    action = state + 1
                    expected.append((state.copy(), action.copy()))
                    camera_values = [40 + episode * 20 + index + camera * 80 for camera in range(2)]
                    expected_images.append(camera_values)
                    frame_images = {key: np.full_like(image, camera_values[camera])
                                    for camera, (key, image) in enumerate(images.items())}
                    takes.add({"observation.state": state, "action": action, "task": "put cube in box", **frame_images})
                before_stop = time.perf_counter()
                takes.space()
                stop_times.append(time.perf_counter() - before_stop)
                if episode == 0:
                    self.assertTrue(saving.wait(timeout=1))
            self.assertEqual(takes.saved, 0)  # Both captures finished while the first save was blocked.
            release_save.set()
            takes.flush()
            dataset.save_episode = original_save
            loaded = LeRobotDataset("local/test", root=root)
            self.assertEqual(loaded.num_episodes, 2)
            self.assertEqual(loaded.num_frames, 10)
            self.assertEqual(len(list(root.rglob("*.mp4"))), 4)  # two episodes, two cameras
            self.assertEqual(sorted(p.name for p in (root / "data").rglob("*.parquet")), ["file-000.parquet", "file-001.parquet"])
            for camera in images:
                self.assertEqual(sorted(p.name for p in (root / "videos" / camera).rglob("*.mp4")), ["file-000.mp4", "file-001.mp4"])
            for index, (state, action) in enumerate(expected):
                row = loaded[index]
                np.testing.assert_array_equal(row["observation.state"].numpy(), state)
                np.testing.assert_array_equal(row["action"].numpy(), action)
                self.assertEqual(row["task"], "put cube in box")
                self.assertAlmostEqual(row["timestamp"].item(), (index % 5) / 20, places=6)
                self.assertEqual(tuple(row["observation.images.camera_0"].shape), (3, 48, 64))
                for camera, key in enumerate(images):
                    self.assertAlmostEqual(float(row[key].mean()) * 255, expected_images[index][camera], delta=3)
            audit = {
                "episodes": loaded.num_episodes, "frames": loaded.num_frames,
                "state_action_sha256": hashlib.sha256(b"".join(s.tobytes() + a.tobytes() for s, a in expected)).hexdigest(),
                "checked_state_action_rows": len(expected), "checked_camera_frames": len(expected) * 2,
                "max_stop_handoff_ms": round(max(stop_times) * 1000, 3),
            }
            print("Round-trip audit:", json.dumps(audit, sort_keys=True))
            takes.discard()  # Delete the last saved episode and preserve the first.
            retained = LeRobotDataset("local/test", root=root)
            self.assertEqual(retained.num_episodes, 1)
            self.assertEqual(retained.num_frames, 5)
            np.testing.assert_array_equal(retained[0]["action"].numpy(), expected[0][1])
            self.assertEqual(len(json.loads((root / "meta/recording/episode-000000.json").read_text())["capture_time_ns"]), 5)
            self.assertFalse((root / "meta/recording/episode-000001.json").exists())
            takes.discard()  # Delete the only remaining episode, then record again.
            self.assertEqual(takes.saved, 0)
            takes.space()
            for _ in range(5):
                takes.add({"observation.state": np.zeros(8, np.float32), "action": np.ones(8, np.float32),
                           "task": "new take after deletion", **images})
            takes.space()
            # The dataset can be opened while the CLI is waiting for the next episode.
            takes.flush()
            replacement = LeRobotDataset("local/test", root=root)
            self.assertEqual(replacement.num_episodes, 1)
            self.assertEqual(replacement[0]["task"], "new take after deletion")
            self.assertEqual([p.name for p in (root / "data").rglob("*.parquet")], ["file-000.parquet"])
            takes.finish()


class CameraWritesTest(unittest.TestCase):
    def test_more_than_64_blocked_image_writes_do_not_stop_capture(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        images = {f"observation.images.camera_{i}": np.full((16, 16, 3), i * 50, np.uint8) for i in range(2)}
        release = threading.Event()
        original_write = EpisodeCapture.write_image

        def stalled_write(image, path):
            if not release.wait(timeout=5):
                raise RuntimeError("Image writer was never released")
            original_write(image, path)

        with tempfile.TemporaryDirectory(prefix="norma-image-backlog-") as temp:
            dataset = LeRobotDataset.create(
                repo_id="local/write-test", root=Path(temp) / "dataset", fps=20,
                features=features_for(images, [f"motor_{i}.pos" for i in range(8)]), vcodec="h264",
            )
            capture = EpisodeCapture(dataset, 0)
            try:
                with patch.object(EpisodeCapture, "write_image", staticmethod(stalled_write)):
                    for i in range(80):
                        capture.add({"observation.state": np.full(8, i, np.float32),
                                     "action": np.full(8, i, np.float32), "task": "backlog test", **images})
                    self.assertEqual(capture.buffer["size"], 80)
                    self.assertEqual(sum(not f.done() for f in capture.pending), 160)
                    release.set()
                    capture.finish()
                self.assertEqual(len(list(dataset.root.rglob("*.png"))), 160)
                self.assertEqual(len(capture.capture_times_ns), 80)
            finally:
                release.set()
                capture.discard()
                dataset.finalize()

    def test_native_x264_statistics_stay_silent_after_callback_reset(self):
        script = """\
import av
from record import quiet_video_logs
quiet_video_logs()
av.logging.restore_default_callback()
av.logging.log(av.logging.INFO, 'libx264', 'HIDDEN_ENCODER_STATS\\n')
av.logging.log(av.logging.ERROR, 'test', 'VISIBLE_REAL_ERROR\\n')
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertNotIn("HIDDEN_ENCODER_STATS", result.stderr)
        self.assertIn("VISIBLE_REAL_ERROR", result.stderr)


class CancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_ctrl_c_waits_for_disk_mutation(self):
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()

        def mutation():
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            finished.set()

        task = asyncio.create_task(blocking(mutation))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())


class ResumeTest(unittest.TestCase):
    def test_resume_empty_then_append_preserves_earlier_files(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        images = {f"observation.images.camera_{i}": np.full((48, 64, 3), i * 50, np.uint8) for i in range(2)}
        features = features_for(images, [f"motor_{i}.pos" for i in range(1, 9)])
        provenance = {"leader_bus": "leader", "follower_bus": "follower", "fps": 20,
                      "cameras": {"camera_0": "front", "camera_1": "wrist"},
                      "calibration_ranges": {"leader": [[100, 1100]] * 8, "follower": [[100, 1100]] * 8},
                      "motor_ids": list(range(1, 9)), "units": "normalized", "action": "absolute"}

        def open_at(root, fps=20, spec=features, collection=provenance, remap=False):
            return open_dataset(root, "local/resume-test", fps, "elrobot", spec, collection, remap)

        def record(takes, value):
            takes.space()
            for _ in range(5):
                takes.add({"observation.state": np.full(8, value, np.float32),
                           "action": np.full(8, value + 1, np.float32), "task": "resume test", **images})
            takes.space()
            takes.finish()

        def hashes(root):
            return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*") if p.is_file()}

        with tempfile.TemporaryDirectory(prefix="norma-resume-test-") as temp:
            root = Path(temp) / "dataset"
            empty = Takes(open_at(root))
            empty.finish()
            first = Takes(open_at(root))  # Resume a launch that quit without recording.
            self.assertEqual(first.saved, 0)
            record(first, 1)
            before = {p: digest for p, digest in hashes(root).items() if p.startswith(("data/", "videos/"))}
            # An interrupted next take must not block resume or mix with new images.
            interrupted = root / "images/observation.images.camera_0/episode-000001/frame-009999.png"
            interrupted.parent.mkdir(parents=True, exist_ok=True)
            interrupted.write_bytes(b"preserve interrupted image bytes")
            second = Takes(open_at(root))
            self.assertEqual(second.saved, 1)
            recovered = list((root.parent / "dataset-recovery").rglob("frame-009999.png"))
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0].read_bytes(), b"preserve interrupted image bytes")
            self.assertFalse(interrupted.exists())
            record(second, 2)
            after = hashes(root)
            for path, digest in before.items():
                self.assertEqual(after[path], digest, f"Resume changed {path}")
            self.assertEqual(sorted(p.name for p in (root / "data").rglob("*.parquet")), ["file-000.parquet", "file-001.parquet"])
            loaded = LeRobotDataset("local/resume-test", root=root)
            self.assertEqual(loaded.num_episodes, 2)
            self.assertEqual(loaded.num_frames, 10)
            for i in range(10):
                row = loaded[i]
                np.testing.assert_array_equal(row["observation.state"].numpy(), np.full(8, 1 if i < 5 else 2))
                for key in images:
                    self.assertEqual(tuple(row[key].shape), (3, 48, 64))
            third = Takes(open_at(root))
            self.assertEqual(third.saved, 2)
            third.finish()
            self.assertEqual(len(list((root.parent / "dataset-recovery").iterdir())), 1)
            snapshot = hashes(root)
            for kwargs in ({"fps": 10}, {"collection": {**provenance, "cameras": {"camera_0": "changed"}}},
                           {"spec": {**features, "observation.state": {**features["observation.state"], "shape": (7,)}}}):
                with self.assertRaisesRegex(ValueError, "Cannot resume"):
                    open_at(root, **kwargs)
                self.assertEqual(hashes(root), snapshot)

            changed = {**provenance, "cameras": {"camera_0": "new-front", "camera_1": "new-wrist"}}
            # Remapping must still reject changed view names and image shapes without mutations.
            for kwargs in (
                {"collection": {**changed, "cameras": {"other": "new-front"}}},
                {"collection": changed, "spec": {**features, "observation.state": {**features["observation.state"], "shape": (7,)}}},
            ):
                with self.assertRaisesRegex(ValueError, "Cannot"):
                    open_at(root, remap=True, **kwargs)
                self.assertEqual(hashes(root), snapshot)
            remapped = Takes(open_at(root, collection=changed, remap=True))
            self.assertEqual(remapped.saved, 2)
            record(remapped, 3)
            saved = json.loads((root / "meta/collection.json").read_text())
            self.assertEqual(saved["cameras"], changed["cameras"])
            self.assertEqual(saved["camera_remaps"][0]["previous"], provenance["cameras"])
            self.assertEqual(saved["camera_remaps"][0]["from_episode"], 2)
            for path, digest in before.items():
                self.assertEqual(hashes(root)[path], digest)
            resumed = Takes(open_at(root, collection=changed))
            self.assertEqual(resumed.saved, 3)
            resumed.finish()

    def test_unknown_folder_and_concurrent_recorder_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="norma-resume-guard-") as temp:
            root = Path(temp)
            sentinel = root / "keep.txt"
            sentinel.write_text("existing unrelated data")
            with self.assertRaisesRegex(ValueError, "expected a dataset"):
                open_dataset(root, "local/test", 20, "elrobot", {}, {})
            self.assertEqual(sentinel.read_text(), "existing unrelated data")
            with dataset_lock(root):
                with self.assertRaisesRegex(ValueError, "Another recorder"):
                    with dataset_lock(root):
                        self.fail("Second recorder acquired the same dataset")


if __name__ == "__main__":
    unittest.main()
