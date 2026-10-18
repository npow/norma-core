"""Read-only discovery, timestamp checks, camera decoding, and existing teleop."""
import asyncio
import io
import logging
import time

import numpy as np
from PIL import Image

from software.station.shared.station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.usbvideo import usbvideo
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
from target.gen_python.protobuf.station import inference

from commands import send_motor_commands, set_torque
from config import MotorConfig
from mirror import ProtectionTable, compute_motor_command, normalize_position
from state import find_bus, parse_motor_state

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class Stream:
    def __init__(self, client, queue, decode):
        self.queue = queue
        self.decode = decode
        self.latest = None
        self.received = 0.0
        self.error = None
        self.entries = asyncio.Queue()
        self.errors = client.follow(queue, self.entries)
        self.task = asyncio.create_task(self.run())

    async def run(self):
        previous = None
        try:
            while True:
                if not self.errors.empty():
                    raise RuntimeError(str(self.errors.get_nowait()))
                try:
                    entry = await asyncio.wait_for(self.entries.get(), 0.1)
                except TimeoutError:
                    continue
                if entry is None:
                    raise RuntimeError("stream closed")
                entry_id = bytes(entry.ID.ID)
                if entry_id == previous:
                    continue
                previous = entry_id
                decoded = self.decode(entry.Data)
                if decoded is not None:
                    self.latest = decoded
                    self.received = time.monotonic()
        except Exception as exc:
            self.error = f"{self.queue}: {exc}"

    def fresh(self, max_age):
        if self.error:
            raise RuntimeError(self.error)
        if self.latest is None or time.monotonic() - self.received > max_age:
            raise RuntimeError(f"No fresh data from {self.queue}")
        return self.latest


def decode_camera(data):
    envelope = usbvideo.RxEnvelopeReader(data)
    pack = envelope.get_frames()
    if pack is None:
        return None
    stamps = pack.get_stamps() or []
    fmt = pack.get_format()
    if not stamps or fmt is None:
        return None
    frames = pack.get_frames_data() or []
    if fmt.get_kind() == 1:
        if not frames:
            return None
        image = np.asarray(Image.open(io.BytesIO(bytes(frames[-1]))).convert("RGB"))
    else:
        width, height = fmt.get_width(), fmt.get_height()
        size = 3 * width * height
        raw = bytes(frames[-1]) if frames else bytes(pack.get_linear_data())[-size:]
        image = np.frombuffer(raw, dtype=np.uint8).reshape(3, height, width).transpose(1, 2, 0).copy()
    stamp = stamps[-1]
    return image, stamp.get_local_stamp_ns(), stamp.get_app_start_id()


def normalized(position, motor):
    percentage = normalize_position(position, motor.range_min, motor.range_max, MotorConfig())
    return percentage  # gripper uses [0, 100]; joints converted below


def vector(positions, motors):
    result = [normalized(p, m) for p, m in zip(positions, motors, strict=True)]
    return np.asarray([2 * p - 100 if i < len(result) - 1 else p for i, p in enumerate(result)], dtype=np.float32)


class Station:
    def __init__(self, client):
        self.client = client
        self.motors = Stream(client, "st3215/inference", st3215.InferenceStateReader)
        self.index = Stream(client, "inference-states", inference.InferenceRxReader)
        self.cameras = {}
        self.config = MotorConfig()
        self.protection = ProtectionTable()
        self.motor_ids = ()
        self.other_controller = None

    def configure(self, leader, follower):
        buses = [find_bus(self.motors.latest, serial) for serial in (leader, follower)]
        if any(bus is None for bus in buses):
            raise RuntimeError("Leader or follower bus missing")
        ids = [tuple(sorted(m.get_id() for m in bus.get_motors() or [])) for bus in buses]
        if ids[0] != ids[1] or ids[0] not in (tuple(range(1, 7)), tuple(range(1, 9))):
            raise RuntimeError(f"Need matching SO101 (1–6) or ElRobot (1–8) motor IDs; got {ids}")
        self.motor_ids = ids[0]
        self.robot_type = "so101_follower" if len(self.motor_ids) == 6 else "elrobot"
        self.joint_names = JOINT_NAMES if len(self.motor_ids) == 6 else [f"motor_{i}.pos" for i in self.motor_ids]

    @classmethod
    async def connect(cls, server):
        return cls(await new_station_client(server, logging.getLogger("station")))

    def configure_follower(self, follower, motor_ids):
        """Policy rollout needs a calibrated follower, independently of a leader."""
        bus = find_bus(self.motors.latest, follower)
        if bus is None:
            raise RuntimeError(f"Follower bus {follower} missing")
        ids = tuple(sorted(m.get_id() for m in bus.get_motors() or []))
        if ids != tuple(motor_ids) or ids not in (tuple(range(1, 7)), tuple(range(1, 9))):
            raise RuntimeError(f"Follower motor IDs differ from training: {ids}")
        self.motor_ids = ids
        self.robot_type = "so101_follower" if len(ids) == 6 else "elrobot"
        self.joint_names = JOINT_NAMES if len(ids) == 6 else [f"motor_{i}.pos" for i in ids]

    async def discover(self, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.index.latest:
                for entry in self.index.latest.get_entries() or []:
                    if entry.get_type() == 32 and self.other_controller is None:
                        self.other_controller = Stream(self.client, entry.get_queue(), mirroring.RxEnvelopeReader)
                    if entry.get_type() == 21:
                        queue = entry.get_queue()
                        if "/" not in queue:
                            queue = "usbvideo/" + queue
                        if queue not in self.cameras:
                            self.cameras[queue] = Stream(self.client, queue, decode_camera)
            if self.motors.latest and self.cameras and all(s.latest for s in self.cameras.values()):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("Discovery timed out: need live arm state and at least one Station camera")

    def describe(self):
        for bus in self.motors.latest.get_buses() or []:
            print(f"Bus {bus.get_bus().get_serial_number()}: "
                  f"motors {[m.get_id() for m in bus.get_motors() or []]}")
        for name, stream in sorted(self.cameras.items()):
            if stream.latest:
                print(f"Camera {name}: {stream.latest[0].shape}")

    def snapshot(self, leader, follower, cameras, max_age):
        if self.other_controller and self.other_controller.latest:
            state = self.other_controller.latest.get_state()
            for mirror in state.get_mirroring() or []:
                ids = [mirror.get_source().get_id().get_unique_id()]
                ids += [t.get_id().get_unique_id() for t in mirror.get_targets() or []]
                if leader in ids or follower in ids:
                    raise RuntimeError("Station mirroring is already active for these arms; stop it in the app first")
        frame = self.motors.fresh(max_age)
        motor_sets = []
        calibration = {}
        now = time.time_ns()
        source_times = []
        for serial in ((leader, follower) if leader is not None else (follower,)):
            bus = find_bus(frame, serial)
            if bus is None:
                raise RuntimeError(f"Bus {serial} missing")
            readers = {m.get_id(): m for m in bus.get_motors() or []}
            if set(readers) != set(self.motor_ids):
                raise RuntimeError(f"Bus {serial}: motor IDs changed to {sorted(readers)}")
            motors = []
            for mid in self.motor_ids:
                reader = readers[mid]
                stamp = reader.get_system_stamp_ns()
                if abs(now - stamp) > max_age * 1e9:
                    raise RuntimeError(f"Bus {serial} motor {mid}: stale source timestamp")
                source_times.append(stamp)
                error = reader.get_error()
                # Gremlin returns an empty reader even when the message is absent.
                has_error = error.get_kind() != 0 or bool(error.get_description()) or bool(error.get_servo())
                if has_error:
                    detail = error.get_description() or f"motor error kind {error.get_kind()}"
                    raise RuntimeError(f"Bus {serial} motor {mid}: {detail}. Check its USB connection in Station.")
                if len(reader.get_state()) < 0x47:
                    raise RuntimeError(f"Bus {serial} motor {mid}: incomplete motor telemetry; check its USB connection in Station")
                if not reader.get_range_freezed():
                    raise RuntimeError(f"Bus {serial} motor {mid}: calibrate and freeze first")
                motor = parse_motor_state(reader)
                if motor.range_min == motor.range_max:
                    raise RuntimeError(f"Bus {serial} motor {mid}: zero calibration range")
                if motor.error_status & (~1 if serial == leader else 255):
                    raise RuntimeError(f"Bus {serial} motor {mid}: motor error")
                motors.append(motor)
            motor_sets.append(motors)
            calibration[serial] = [[m.range_min, m.range_max] for m in motors]
        images = {}
        for name, queue in cameras.items():
            image, stamp, _ = self.cameras[queue].fresh(max_age)
            if abs(now - stamp) > max_age * 1e9:
                raise RuntimeError(f"Camera {name}: stale source timestamp")
            source_times.append(stamp)
            images[f"observation.images.{name}"] = image
        if max(source_times) - min(source_times) > max_age * 1e9:
            raise RuntimeError("Camera/motor source timestamps exceed allowed skew")
        source, target = motor_sets if leader is not None else (None, motor_sets[0])
        return source, target, images, calibration

    async def enable(self, leader, follower):
        await set_torque(self.client, leader, self.motor_ids, enable=False)
        await set_torque(self.client, follower, self.motor_ids, enable=True)

    async def disable(self, follower):
        await set_torque(self.client, follower, self.motor_ids, enable=False)

    async def step(self, leader_motors, follower_motors, follower):
        commands = []
        goals = [m.target_position for m in follower_motors]
        for i, (source, target) in enumerate(zip(leader_motors, follower_motors, strict=True)):
            cmd = compute_motor_command(i + 1, source, target, self.protection.get(i + 1), self.config)
            if cmd:
                commands.append(cmd)
                if cmd.goal is not None:
                    goals[i] = cmd.goal
        await send_motor_commands(self.client, follower, commands)
        return vector(goals, follower_motors)
