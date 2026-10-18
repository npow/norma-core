"""Takeover, capture, and training-export checks without robot connections."""
import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from bootstrap import setup
setup()
from mirror import MotorState
from intervention import GrabDetector, RolloutRecorder, LeaderMirror
from export_interventions import export, selected_segments
from record import features_for


class GrabTests(unittest.TestCase):
    def test_sustained_effort_and_error_same_joint_distinct_frames(self):
        detector = GrabDetector(seconds=.15, grace=.1)
        motors = [MotorState(present_position=2000, target_position=2100, current=50)]
        self.assertFalse(detector.update(.05, 1, motors))
        self.assertFalse(detector.update(.1, 2, motors))
        self.assertFalse(detector.update(.4, 2, motors))
        self.assertTrue(detector.update(.4, 3, motors))
        detector.reset(.4)
        self.assertFalse(detector.update(.45, 4, motors))

    def test_noise_wrap_and_different_joints_do_not_trigger(self):
        detector = GrabDetector(grace=0)
        motors = [MotorState(present_position=4090, target_position=10, current=100),
                  MotorState(present_position=1000, target_position=1200, current=0)]
        self.assertFalse(detector.update(1, 1, motors))
        self.assertFalse(detector.update(2, 2, motors))
        motors[1].current = 50
        self.assertFalse(detector.update(3, 3, motors))
        motors[1].current = 0
        self.assertFalse(detector.update(3.1, 4, motors))
        motors[1].current = 50
        self.assertFalse(detector.update(3.2, 5, motors))

    def test_full_pose_error_can_trigger_when_slew_limited_goal_is_close(self):
        detector = GrabDetector(grace=0, seconds=.1)
        motors = [MotorState(present_position=2000, target_position=2060, current=50)]
        self.assertFalse(detector.update(1, 1, motors, targets=[2200]))
        self.assertTrue(detector.update(1.2, 2, motors, targets=[2200]))

    def test_export_never_crosses_ownership_or_capture_gaps(self):
        owners = [0, 2, 1, 1, 1, 1, 2, 0, 1]
        times = [0, .05, .1, .15, .4, .45, .5, .55, .6]
        self.assertEqual(list(selected_segments(owners, times, 20)), [(2, 4), (4, 6), (8, 9)])
        self.assertEqual(list(selected_segments(owners, times, 20, 2)), [(2, 4), (4, 6)])


def recording_features(images, names):
    features = features_for(images, names)
    for key in ("policy_action", "leader_state"):
        features[f"intervention.{key}"] = {"dtype": "float32", "shape": (len(names),), "names": names}
    for key in ("owner", "policy_valid", "leader_valid"):
        features[f"intervention.{key}"] = {"dtype": "int64", "shape": (1,), "names": [key]}
    return features


def frame(index, owner):
    return {"observation.state": np.full(6, index, np.float32), "action": np.full(6, index + 1, np.float32),
        "task": "test correction", "observation.images.front": np.full((32, 32, 3), index * 10, np.uint8),
        "intervention.policy_action": np.full(6, index + 2, np.float32),
        "intervention.leader_state": np.full(6, index + 3, np.float32),
        "intervention.owner": np.asarray([owner], np.int64),
        "intervention.policy_valid": np.asarray([owner == 0], np.int64),
        "intervention.leader_valid": np.asarray([1], np.int64)}


class CaptureTests(unittest.TestCase):
    def test_video_labels_resume_partial_and_human_export(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "rollouts"
            features = recording_features({"observation.images.front": np.zeros((32, 32, 3), np.uint8)}, [f"joint_{i}" for i in range(6)])
            provenance = {"leader_bus": "leader", "follower_bus": "follower", "cameras": {"front": "camera"},
                "calibration_ranges": {}, "motor_ids": list(range(1, 7)), "units": "test", "action": "protected goals"}
            recorder = RolloutRecorder(root, "local/test", 20, "so101_follower", features, provenance)
            recorder.begin({"checkpoint": "test"})
            owners = [0, 2, 1, 1, 2, 0]
            for i, owner in enumerate(owners):
                recorder.add(frame(i, owner), {"owner": ["policy", "human", "transition"][owner], "time_s": i / 20})
            recorder.end("success", "success")
            recorder.close()
            loaded = LeRobotDataset("local/test", root=root, video_backend="pyav")
            self.assertEqual(loaded.num_frames, 6)
            self.assertEqual(loaded.hf_dataset[2]["intervention.owner"].item(), 1)
            self.assertTrue(np.allclose(loaded[3]["action"].numpy(), 4))
            self.assertAlmostEqual(float(loaded[3]["observation.images.front"].mean()), 30 / 255, delta=.02)
            audit = json.loads((root / "meta/interventions/episode-000000.json").read_text())
            self.assertEqual(audit["control_counts"], {"policy": 2, "human": 2, "transition": 2})
            self.assertEqual(len(audit["frame_sha256"]), 64)
            self.assertEqual(audit["outcome"], "success")
            recorder = RolloutRecorder(root, "local/test", 20, "so101_follower", features, provenance)
            recorder.begin({"checkpoint": "test-2"})
            recorder.add(frame(6, 0), {"owner": "policy", "time_s": 0})
            recorder.close()
            audit2 = json.loads((root / "meta/interventions/episode-000001.json").read_text())
            self.assertTrue(audit2["incomplete"])
            manifest = export(root, Path(temp) / "corrections", minimum=2)
            self.assertEqual(manifest["total_frames"], 2)
            corrections = LeRobotDataset("local/human-corrections", root=Path(temp) / "corrections", video_backend="pyav")
            self.assertEqual(corrections.num_episodes, 1)
            self.assertTrue(np.allclose(corrections[0]["action"].numpy(), 3))
            self.assertTrue(np.allclose(corrections[1]["action"].numpy(), 4))
            self.assertNotIn("intervention.owner", corrections.features)
            # The export is frozen against mutations of saved rollout artifacts.
            journal = root / "meta/interventions/episode-000000.jsonl"
            journal.write_text(journal.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                export(root, Path(temp) / "tampered")
            print(f"AUDIT: 7/7 rollout frames, 2/2 exported human frames, SHA256={audit['frame_sha256']}")

    def test_queue_full_is_reported_without_blocking(self):
        import queue
        recorder = object.__new__(RolloutRecorder)
        recorder.queue = queue.Queue(maxsize=1)
        recorder.queue.put(("frame", None))
        recorder.error, recorder.closed = None, False
        with self.assertRaisesRegex(RuntimeError, "backlog full"):
            recorder.capacity()
        with self.assertRaisesRegex(RuntimeError, "backlog full"):
            recorder.put("frame", None)

    def test_invalid_open_reports_error_without_hanging(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "recording failed"):
                RolloutRecorder(temp, "local/test", 20, "test", {}, {})


class MirrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_goals_seeded_before_enable_and_original_limit_restored(self):
        from types import SimpleNamespace
        state = bytearray(0x47)
        state[0x30:0x32] = (75).to_bytes(2, "little")
        reader = SimpleNamespace(get_id=lambda: 1, get_state=lambda: state)
        bus = SimpleNamespace(get_motors=lambda: [reader])
        station = SimpleNamespace(client=object(), motor_ids=[1], motors=SimpleNamespace(latest=object()))
        events = []
        async def torque(*args, **kwargs): events.append(("torque", args[-1]))
        async def commands(*args): events.append(("goals", args[-1]))
        async def writes(*args): events.append(("register", args[-1]))
        with patch("state.find_bus", return_value=bus), patch("commands.set_torque", side_effect=torque), \
             patch("commands.send_motor_commands", side_effect=commands), patch("commands._sync_write", side_effect=lambda *a: a), \
             patch("software.station.shared.station_py.send_commands", side_effect=writes):
            mirror = LeaderMirror(station, "leader", 100, 60)
            await mirror.start([MotorState(present_position=2000)])
            self.assertEqual([event[0] for event in events], ["torque", "register", "goals", "torque"])
            self.assertEqual(events[1][1][0][2], [(1, (75).to_bytes(2, "little"))])
            self.assertEqual(events[2][1][0].goal, 2000)
            await mirror.restore()
            self.assertEqual(events[-2], ("torque", False))
            self.assertEqual(events[-1][1][0][2], [(1, (75).to_bytes(2, "little"))])

    async def test_failed_disable_does_not_raise_leader_torque_cap(self):
        from types import SimpleNamespace
        mirror = object.__new__(LeaderMirror)
        mirror.touched = True
        mirror.station = SimpleNamespace(client=object())
        mirror.serial, mirror.original = "leader", [(1, b"\xf4\x01")]
        mirror.stop = AsyncMock(side_effect=RuntimeError("disconnect"))
        with patch("software.station.shared.station_py.send_commands", new=AsyncMock()) as send:
            with self.assertRaisesRegex(RuntimeError, "disconnect"):
                await mirror.restore()
            send.assert_not_awaited()


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, execute=True, fail_recording=False, auto=False, delayed=False):
        from types import SimpleNamespace
        import rollout_act
        import station_io
        from config import MotorConfig
        from mirror import ProtectionTable
        keys = asyncio.Queue()
        keys.put_nowait(" ")
        calibration = [[1000, 3000]] * 6
        image = np.zeros((32, 32, 3), np.uint8)
        collection = {"leader_bus": "leader", "follower_bus": "follower", "cameras": {"front": "camera"},
            "fps": 20, "motor_ids": list(range(1, 7)), "calibration_ranges": {"leader": calibration, "follower": calibration},
            "task": "test", "units": "test"}
        metadata = {"collection": collection, "checkpoint": "test", "chunk_size": 10,
            "features": {"observation.images.front": {"shape": [32, 32, 3]}}}
        motors = [MotorState(present_position=2000, target_position=2000, range_min=1000, range_max=3000) for _ in range(6)]
        observations = [0]
        recorder_events, writes, mirror_events = [], [], []
        def snapshot(*args):
            observations[0] += 1
            number = observations[0]
            if number == 6 and not auto: keys.put_nowait("t")
            if number == 11: keys.put_nowait("r")
            if number == 22:
                keys.put_nowait("s")
                keys.put_nowait("q")
            return motors, motors, {"observation.images.front": image}, collection["calibration_ranges"]
        station = SimpleNamespace(discover=AsyncMock(), configure=lambda *a: None,
            configure_follower=lambda *a: None, motor_ids=list(range(1, 7)), config=MotorConfig(),
            protection=ProtectionTable(), robot_type="so101_follower", joint_names=[f"joint_{i}" for i in range(6)],
            snapshot=snapshot, client=SimpleNamespace(close_connection=lambda: writes.append("closed")),
            disable=AsyncMock(side_effect=lambda *a: writes.append("disable")),
            motors=SimpleNamespace(latest=object()), cameras={"camera": SimpleNamespace(latest=(image, 123, 0))})
        readers = [SimpleNamespace(get_system_stamp_ns=lambda: observations[0]) for _ in range(6)]
        bus = SimpleNamespace(get_motors=lambda: readers)
        class Mirror:
            active = False
            def __init__(self, *a): pass
            async def start(self, *a): self.active = True; mirror_events.append("start")
            async def stop(self): self.active = False; mirror_events.append("stop")
            async def restore(self): self.active = False; mirror_events.append("restore")
            async def step(self, *a): mirror_events.append("step")
        class Recorder:
            def __init__(self, root, *a, **kw): self.root = root
            def wait_ready(self): pass
            def check(self):
                if fail_recording and observations[0] >= 8: raise RuntimeError("disk full")
            def capacity(self): self.check()
            def begin(self, data): recorder_events.append(("begin", data))
            def add(self, frame, detail): recorder_events.append(("frame", frame, detail))
            def end(self, reason, outcome): recorder_events.append(("end", reason, outcome))
            def close(self): recorder_events.append(("close",))
        @contextlib.contextmanager
        def keyboard(): yield keys
        prediction_calls = [0]
        async def predict(*args):
            prediction_calls[0] += 1
            if delayed and prediction_calls[0] == 2:
                try:
                    await asyncio.sleep(2)
                except asyncio.CancelledError:
                    writes.append("prediction_cancelled")
                    raise
            await asyncio.sleep(0)
            return args[1], {"actions": [[10.] * 5 + [55.]] * 10}, 1
        async def send(*args): writes.append("send")
        async def torque(*args, **kwargs): writes.append("enable")
        with tempfile.TemporaryDirectory() as temp, patch("sys.argv", ["rollout_act.py", "--teleop-takeover",
                "--handoff-seconds", ".01", "--dataset-root", temp, "--seconds", "10",
                "--chunk-steps", "2", "--diagnostics-jsonl", str(Path(temp) / "diagnostics.jsonl"),
                *(["--execute"] if execute else [])]):
            args = rollout_act.parse_args()
            with patch.object(station_io.Station, "connect", new=AsyncMock(return_value=station)), \
                 patch("record.keyboard", keyboard), patch("rollout_act.request_json", return_value=metadata), \
                 patch("rollout_act.predict_async", side_effect=predict), patch("commands.send_motor_commands", side_effect=send), \
                 patch("commands.set_torque", side_effect=torque), patch("intervention.LeaderMirror", Mirror), \
                 patch("intervention.RolloutRecorder", Recorder), patch("state.find_bus", return_value=bus), \
                 patch("intervention.GrabDetector.update", side_effect=lambda *a, **kw: auto and observations[0] == 7):
                if fail_recording:
                    with self.assertRaisesRegex(RuntimeError, "disk full"):
                        await asyncio.wait_for(rollout_act.run(args), 4)
                else:
                    await asyncio.wait_for(rollout_act.run(args), 4)
        return recorder_events, writes, mirror_events

    async def test_policy_takeover_resume_and_recorded_actions(self):
        records, writes, mirrors = await self.exercise()
        frames = [row for row in records if row[0] == "frame"]
        self.assertIn("human", {row[2]["owner"] for row in frames})
        human_index = next(i for i, row in enumerate(frames) if row[2]["owner"] == "human")
        self.assertTrue(any(row[2]["owner"] == "policy" for row in frames[human_index + 1:]))
        for _, frame_data, detail in frames:
            self.assertEqual(frame_data["action"].tolist(), detail["protected_goal"])
        self.assertEqual(mirrors.count("start"), 2)
        self.assertEqual(mirrors[-1], "restore")
        self.assertIn(("end", "success", "success"), records)
        self.assertEqual(records[-1], ("close",))
        self.assertEqual(writes[-1], "closed")

    async def test_pending_policy_response_is_cancelled_at_takeover(self):
        records, writes, mirrors = await self.exercise(delayed=True)
        self.assertIn("prediction_cancelled", writes)
        human = [row for row in records if row[0] == "frame" and row[2]["owner"] == "human"]
        self.assertTrue(human)
        self.assertTrue(all(row[2]["predicted_action"] is None for row in human))

    async def test_preview_has_zero_writes_and_no_training_frames(self):
        records, writes, mirrors = await self.exercise(execute=False)
        self.assertEqual(records, [])
        self.assertEqual(writes, ["closed"])
        self.assertEqual(mirrors, [])

    async def test_recording_failure_disables_motion_before_closing(self):
        records, writes, mirrors = await self.exercise(fail_recording=True)
        self.assertIn("disable", writes)
        self.assertEqual(mirrors[-1], "restore")
        self.assertEqual(records[-1], ("close",))
        self.assertEqual(writes[-1], "closed")

    async def test_automatic_grab_also_latches_human_control(self):
        records, writes, mirrors = await self.exercise(auto=True)
        self.assertIn("human", {row[2]["owner"] for row in records if row[0] == "frame"})


if __name__ == "__main__":
    unittest.main()
