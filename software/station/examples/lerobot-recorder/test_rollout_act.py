"""Verify action units and hardware protections without connecting to Station."""
import unittest

from bootstrap import setup
setup()
from config import MotorConfig
from mirror import MotorState, ProtectionTable
from rollout_act import protected_commands, TimedTrajectory


class DiagnosticWriterTests(unittest.TestCase):
    def test_blocked_disk_write_does_not_block_control_thread(self):
        import threading
        from rollout_act import DiagnosticWriter
        started, release = threading.Event(), threading.Event()
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def write(self, text):
                started.set()
                if not release.wait(2):
                    raise RuntimeError("Test writer timed out")
        class Path:
            def open(self, *args, **kwargs): return Stream()
        writer = DiagnosticWriter(Path())
        try:
            writer.write("first")
            self.assertTrue(started.wait(1))
            # This enqueue completes while the worker is blocked writing first.
            writer.write("second")
            self.assertEqual(writer.queue.qsize(), 1)
        finally:
            release.set()
            writer.close()


class ChunkTimingTests(unittest.TestCase):
    def test_late_startup_chunk_is_rejected_even_before_horizon_expires(self):
        from rollout_act import response_is_fresh
        self.assertFalse(response_is_fresh(100, 102.044, 20, 50, 10))
        self.assertFalse(response_is_fresh(100, 100.501, 20, 50, 10))
        self.assertTrue(response_is_fresh(103, 103.1, 20, 50, 10))
        self.assertFalse(response_is_fresh(100, 102.5, 20, 50, 10))

    def test_observation_timestamp_skips_inference_delay(self):
        chunk = TimedTrajectory(100, 20, [[float(i), 50.] for i in range(50)])
        self.assertAlmostEqual(chunk.action_at(100.25)[0], 5.)
        self.assertAlmostEqual(chunk.action_at(100.275)[0], 5.5)

    def test_expired_chunks_stop(self):
        chunk = TimedTrajectory(100, 20, [[0., 50.]] * 50)
        with self.assertRaises(RuntimeError):
            chunk.action_at(102.5)

    def test_hermite_splice_matches_position_and_velocity(self):
        old = TimedTrajectory(100, 20, [[float(i), 0.] for i in range(50)])
        new = TimedTrajectory(100.2, 20, [[100. + 2 * i, 100.] for i in range(50)])
        now = 100.5
        old_position, old_velocity = old.action_at(now)[0], old.velocity_at(now)[0]
        new.splice_from(old, now, .15)
        self.assertAlmostEqual(new.action_at(now)[0], old_position)
        self.assertAlmostEqual(new.velocity_at(now)[0], old_velocity)
        end = now + .15
        self.assertAlmostEqual(new.action_at(end)[0], new.raw_action_at(end)[0])
        self.assertAlmostEqual(new.velocity_at(end)[0], 40.)
        self.assertEqual(new.action_at(now)[-1], 100.)

    def test_actions_keep_advancing_while_next_chunk_is_pending(self):
        chunk = TimedTrajectory(0, 20, [[float(i), 50.] for i in range(50)])
        self.assertEqual([round(chunk.action_at(t)[0]) for t in [.5, .55, .6, .65]], [10, 11, 12, 13])


class ActionMappingTests(unittest.TestCase):
    def setUp(self):
        self.motors = [MotorState(present_position=2000, target_position=2000,
                                  range_min=1000, range_max=3000) for _ in range(8)]
        self.config = MotorConfig()

    def test_absolute_joint_and_gripper_units_without_second_margin(self):
        commands = protected_commands([10] * 7 + [55], self.motors, ProtectionTable(), self.config, 500)
        self.assertEqual([c.goal for c in commands], [2100] * 8)

    def test_step_limit_and_calibrated_end_margin(self):
        commands = protected_commands([1000] * 8, self.motors, ProtectionTable(), self.config, 60)
        self.assertEqual([c.goal for c in commands], [2060] * 8)
        commands = protected_commands([1000] * 8, self.motors, ProtectionTable(), self.config, 3000)
        self.assertEqual([c.goal for c in commands], [2980] * 8)

    def test_trajectory_deadband_tracks_small_changes_without_chasing_one_tick_noise(self):
        from dataclasses import replace
        tracking = replace(self.config, deadband=2)
        self.assertEqual(protected_commands([.1] * 7 + [50.05], self.motors, ProtectionTable(), tracking, 60), [])
        commands = protected_commands([.8] * 7 + [50.4], self.motors, ProtectionTable(), tracking, 60)
        self.assertEqual([c.goal for c in commands], [2008] * 8)
        self.assertEqual(protected_commands([.8] * 7 + [50.4], self.motors, ProtectionTable(), self.config, 60), [])

    def test_overcurrent_holds_instead_of_applying_prediction(self):
        self.motors[0].current = 101
        self.motors[0].target_position = 2200
        commands = protected_commands([100] * 7 + [100], self.motors, ProtectionTable(), self.config, 60)
        self.assertEqual(commands[0].goal, 2000)

    def test_nonfinite_or_wrong_dimensions_rejected(self):
        for action in ([0] * 7, [float("nan")] * 8, [float("inf")] * 8):
            with self.assertRaises(ValueError):
                protected_commands(action, self.motors, ProtectionTable(), self.config, 60)

    def test_wrapped_calibration_respects_step_limit(self):
        for motor in self.motors:
            motor.range_min, motor.range_max = 3800, 500
            motor.present_position = motor.target_position = 4080
        commands = protected_commands([100] * 8, self.motors, ProtectionTable(), self.config, 60)
        self.assertEqual([c.goal for c in commands], [44] * 8)


if __name__ == "__main__":
    unittest.main()
