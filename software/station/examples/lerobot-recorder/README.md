# Keyboard demonstration recorder

From the repository root:

```bash
cd software/station/examples/lerobot-recorder
uv run python record.py --task "put the cube in the box" --root ~/datasets/cube-001
```

Station must be running with TCP enabled on port 8888 (the desktop app already does this).
The first `uv run` installs the pinned Python 3.12 environment. Bindings are generated into
`~/.cache/norma-recorder/`; no build or changes to the repository's `target/` are needed.

| Key | Action |
| --- | --- |
| Space | Start a take; press again to stop and queue a background save |
| `d` | Discard the current take, or delete the last saved episode while idle |
| `q` | Queue the final take, wait for background saves, and exit |
| Ctrl-C | Save the partial/final take, wait for background saves, and exit |

Stopping queues the episode for saving and immediately shows `READY — episode 000001`;
press Space to record the next episode while the previous one saves. Each take owns separate
camera buffers and image files, and one background worker writes the episodes in order.
The status shows pending saves; `Saved episode ...` means its files are finalized.
During recording it
shows the current episode number, frame count, and elapsed seconds once per second.
Episode numbering starts at `000000`, matching LeRobot's zero-based episode indexes.
Deleting/discarding reuses that episode number on the next take.
Deleting the last completed take while idle waits for queued saves first. Image writes and
episode saves queue in the background without a backlog limit that stops your take or prevents
the next take from starting. The recording status shows pending image writes. If storage is
persistently slower than capture, queued images use RAM until the writer catches up.
Actual disk/encoder failures are reported; normal x264 encoding statistics are hidden.

Each episode gets its own numbered files: `data/chunk-000/file-000.parquet`, then
`file-001.parquet`, etc., plus matching `file-000.mp4`, `file-001.mp4` for **each** camera
under `videos/observation.images.camera_0/chunk-000/` and `camera_1/chunk-000/`.
The standard LeRobot chunk/file numbering rolls into the next chunk after 1,000 files.
Completed episodes can be opened in the visualizer after their `Saved episode ...` message.
Reusing the same `--root` automatically resumes that dataset. If it contains episodes
`000000` through `000009`, startup shows `READY — episode 000010` and the next Space
records episode 10. Existing state and camera files are preserved. Restart with the same
arms, calibration, camera mapping/resolution, and `--fps`; incompatible settings, incomplete
saved episode files, or unrelated folders are refused. Leftover PNG images from interrupted
takes are moved intact into a timestamped sibling folder such as `cube-001-recovery/` and
recording resumes from the last completed episode. This move does not duplicate image data
or delete it. These images alone cannot reconstruct missing joint states/actions.
A second recorder cannot write the same
dataset at the same time. A dataset created by an earlier launch with zero episodes also resumes.
Use `q` before closing the terminal so every queued save finishes. Abrupt termination/power
loss can leave incomplete takes; auto-resume refuses incomplete files to preserve existing data.

For timed episodes, add `--episode-seconds 30`. Space starts recording, the take stops
and saves automatically after 30 seconds, and the app stays ready for the next Space.
Press Space during recording to stop earlier.

The first Space releases leader torque and enables follower torque. Teleoperation continues
between takes so you can reset the scene. On exit or a data health failure, follower torque is
disabled; support the arm when stopping. Existing Station mirroring for these arms must be stopped
in the app first. The recorder uses the existing Python teleop's calibration mapping, deadband,
speed/acceleration limits, and current protection.

## Your arms and cameras

The configured defaults are leader `5B3D044117`, follower `5B3D045222`, server `localhost`.
Both 8-motor ElRobot and 6-motor SO101 layouts are detected from matching motor IDs.
To inspect hardware without moving anything:

```bash
uv run python record.py --inspect
```

All live Station cameras are recorded by default, ordered by queue ID as `camera_0`, `camera_1`,
etc. Keep the physical camera positions fixed. To select and name views, copy the full queue
paths printed by `--inspect`:

```bash
uv run python record.py --task "put the cube in the box" --root ~/datasets/cube-002 \
  --camera "front=FULL_QUEUE_PATH" --camera "wrist=FULL_QUEUE_PATH"
```

Use `--leader-bus`, `--follower-bus`, and `--server HOST:PORT` for different hardware.
### Resume after unplugging USB cameras

After unplugging cameras, USB queue IDs can change, causing
`Cannot resume: cameras differs from the saved collection`.
Run `uv run python record.py --inspect` to list current queue paths, then check the
live views in Station against the saved dataset videos to identify each camera.
For the existing white-lid dataset, `camera_0` is the top view and `camera_1` is the wrist view.
Resume with `--remap-cameras`
and explicit `--camera camera_0=NEW_QUEUE_PATH --camera camera_1=NEW_QUEUE_PATH`
assignments for every saved view. Match each name to the same physical view as before;
queue sort order may have changed. Camera names and image shapes must still match.
The recorder retains previous IDs and the episode where the mapping changed in
`meta/collection.json`; existing episodes are preserved. Use these explicit camera
assignments again on subsequent launches.

```bash
uv run python record.py \
  --task "move the white lid onto the black tray" \
  --root ~/datasets/white-lid \
  --remap-cameras \
  --camera "camera_0=FULL_CURRENT_TOP_QUEUE_PATH" \
  --camera "camera_1=FULL_CURRENT_WRIST_QUEUE_PATH"
```

Replace both queue placeholders with the full paths from `--inspect`. Use the
original task instruction for the dataset. Press **Space** to start/stop a take
and **q** to finish saving and quit. On later launches, keep the `--camera`
assignments; `--remap-cameras` is only needed when those IDs change again.

### Capture timing

Collection defaults to 20 FPS. Slow writes and scheduling hiccups do not end the take.
Actual capture times are saved in `meta/recording/episode-NNNNNN.json` for timing inspection.
Calibration must already be frozen for every motor on both arms. Missing motors, changed
calibration/resolution, stale motor/camera source timestamps, and large sampling overruns stop
teleoperation; the valid portion of the take is queued for saving rather than deleted. Source age and
inter-source skew are bounded by `--max-age` (default 0.2 seconds). A remote Station needs a
synchronized wall clock. This is latest-frame sampling, not hardware-triggered synchronization.

## Dataset and visualization

The writer uses **LeRobot 0.4.4's native v3 API**: Parquet state/action tables, H.264 MP4 videos,
episode/task indexes, and normalization statistics. `meta/collection.json` records the selected
buses, ordered motor IDs, calibration ranges, camera mapping, units, and sampling settings.

`observation.state` contains measured **follower** positions. `action` contains the absolute
follower goals sent after mirroring protection, with retained goals for deadband/no-write ticks.
Observations are sampled before sending the action for that frame. Joint coordinates are
calibrated `[-100, 100]`; the last motor is the gripper, calibrated `[0, 100]`. Six-motor SO101
vectors use the standard joint names; ElRobot vectors use `motor_1.pos` through `motor_8.pos`.
The natural-language `--task` is saved on every frame. Deployment must use the same joint order,
calibration, normalization, and camera mapping.

View locally after quitting the recorder:

```bash
uv run lerobot-dataset-viz --repo-id local/arm-demos \
  --root ~/datasets/cube-001 --episode-index 0 --num-workers 0
```

For the [web dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset), use
an actual Hugging Face repo ID and opt into uploading:

```bash
uv run hf auth login
uv run python record.py --task "put the cube in the box" --root ~/datasets/cube-003 \
  --repo-id YOUR_USERNAME/cube-demos --push-to-hub
```

Uploads happen only after `q` and create a private dataset. The hosted visualizer needs access
to the repo; for private datasets, use the local viewer or explicitly change sharing on the Hub.
Collection without `--push-to-hub` stays local.

## Training

ACT and SmolVLA can use this LeRobot dataset directly through their LeRobot trainers. On your
training machine, copy the complete dataset directory, then start ACT with:

```bash
uv run lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/arm-demos --dataset.root=/path/to/cube-001 \
  --output_dir=outputs/act-cube --batch_size=8 --steps=20000 --wandb.enable=false
```

For SmolVLA, install its model extras in your training environment, use
`--policy.path=lerobot/smolvla_base`, and map your camera keys to the checkpoint's expected
keys with the trainer's `--rename_map` (for example, `camera_0` to `camera1` and `camera_1`
to `camera2`, both under `observation.images.`). The state/action projection supports
padding shorter joint vectors; the 8-motor ElRobot still needs its own fine-tuning and
matching deployment adapter. See the [SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla).
These training runs are examples; the recorder does not start training or rent GPUs.

**Octo-small is not a native LeRobot dataset consumer.** Its standard pipeline uses RLDS.
This dataset preserves images, joint states, actions, tasks, and episode boundaries for
conversion, but an RLDS exporter or a custom Octo loader/action adapter is still needed.
See [Octo's fine-tuning examples](https://github.com/octo-models/octo#octo-finetuning).
The repository's existing custom-Parquet SmolVLA trainer also uses a different format;
use the LeRobot trainer for these datasets.

## Verification

### White-lid ACT rollout

The trained ACT policy runs on `odin`. This Mac sends both camera images and the
calibrated follower state through an SSH tunnel; `odin` returns an action chunk.
The Mac sends protected motor commands through the existing local Station.
The checkpoint stays on `odin`; it does not need to be downloaded to this Mac.

From this directory, preview predictions without moving the robot:

```bash
./run_act_white_lid.sh
```

To enable motor control:

```bash
./run_act_white_lid.sh --execute
```

Press **Space** to start or pause and **q** to quit. Each trial stops after
60 seconds (override with `--seconds 120`); reset the scene and press Space for another trial. Pausing/exiting
releases follower torque, so support the arm. Station must be running locally,
with its mirroring stopped for this follower. Follower calibration must be frozen
and exactly match `meta/collection.json`. A leader is not required for inference.
Camera queue identity, resolution, source age and skew are checked on each tick.

The default is the **5,000-step** checkpoint. A later saved checkpoint can be
selected with, for example:

```bash
ACT_STEP=10000 ./run_act_white_lid.sh --execute
```

The scripts load checkpoint normalization statistics and use the collection's
20 Hz action rate, joint order, absolute joint coordinates `[-100, 100]` and
gripper coordinates `[0, 100]`. They reuse Station's current protection, calibrated
arc limits, and speed/acceleration rules. A default 60-encoder-tick movement limit
per control tick constrains initial model movements. Inference startup, failure,
or disconnection does not bypass observation health checks. Ctrl-C also performs
follower torque cleanup.

`launch_act.py` starts the inference server on `odin` and forwards localhost port
8765. `policy_server.py` loads the model there; `rollout_act.py` runs the local
Station control loop. Closing the launcher closes the tunnel; the warmed worker
exits after 30 minutes without requests.
If `HF_TOKEN` is set in your local shell, the launcher forwards that token through
SSH stdin before Hugging Face initializes. It is not written to disk or included
in SSH command arguments.

SmolVLA uses the same controller and its saved tokenizer/normalization statistics.
The instruction is the exact recorded task, "move the white lid onto the black tray".
To try its downloaded 2,000-step checkpoint:

```bash
./run_smolvla_white_lid.sh --execute
```

It also defaults to 60 seconds. Use `--seconds 120` for longer trials. Stop any
ACT rollout first because both launchers use localhost port 8765. SmolVLA
checkpoints are selected with `SMOLVLA_STEP`, after that checkpoint has been
downloaded from Modal to `odin`.

Check action conversion and protections without moving hardware:

```bash
uv run python -m unittest -v test_rollout_act
```

```bash
uv run python -m unittest -v test_record
```

Tests cover keyboard state transitions, deletion, camera mapping/decoding, normalization,
and an actual two-episode/two-camera LeRobot video write/read round trip. The round trip
checks every state/action row, timestamps, tasks, video frames, and prints a sample hash
with exact denominators. A deliberately blocked save verifies that the next episode can record,
both cameras retain the correct episode/frame contents, and quitting drains the save queue.
`--inspect` separately checks live hardware without sending commands.

SmolVLA now defaults to a persistent `torch.compile(mode="reduce-overhead")`
endpoint on Odin. The first launch compiles and warms the complete sampler before
accepting observations; subsequent launches of that checkpoint reuse the warm
worker. The worker exits after 30 minutes without requests. The SSH tunnel closes
when the rollout exits. `POLICY_BACKEND=eager` selects a separate eager endpoint
for comparison; it never silently substitutes for a failed compile.

```bash
SMOLVLA_STEP=4000 ./run_smolvla_white_lid.sh --execute
# Comparison with the same controller and checkpoint:
POLICY_BACKEND=eager SMOLVLA_STEP=4000 ./run_smolvla_white_lid.sh --execute
```

The controller runs independently of inference at the dataset's 20 Hz. New chunks
are aligned to the observation timestamp and spliced into the current trajectory
with a 150 ms cubic Hermite correction for arm position and velocity. Gripper
commands are excluded from this correction. `--blend-seconds 0` disables blending.
This is a runtime trajectory bridge, not a model trained with RTC or Hermite VLA.

Every trial writes a JSONL file under
`~/benchmarks/white-lid-smolvla-20261017/rollouts/` containing control intervals,
observation age, predicted and protected actions, measured positions, currents,
and model/server/roundtrip timing. Console output includes Hz and
`grip(pred/sent/actual)` to separate policy output from protection or actuation.
Preview mode performs no motor writes. Compilation changes BF16 rounding: six
held-out observations with fixed noise showed mean absolute action drift of
0.10 units, with a maximum gripper difference of 3.85 on its 0–100 scale.

Measured optimized 4k endpoint: 30 live-camera requests, model median 24.6 ms /
p95 27.2 ms, total encoding + network roundtrip median 111 ms / p95 178 ms.
A separate 10-second preview completed 199 control ticks at 20.00 Hz with a
52.12 ms maximum interval and zero motor writes. This measures serving and
controller timing; physical task success remains unverified.

Rollout defaults match the successful executed trial: 20-tick deadband and fresh
sampling noise on each replan. The subsequent 2-tick/fixed-noise defaults regressed
physical approach behavior and were reverted. `--deadband-ticks` and
`--noise-seed` remain explicit experiment options; they are not the default.
Calibration margins, target-step limits, and overcurrent protection remain active.
Control diagnostics now include observation time, motor-send time, and the actual
per-motor goal/speed/acceleration writes. A prior executed trace had five control
gaps above 70 ms (maximum 264 ms); their cause is not established by the old log.

Disk diagnostics and periodic terminal output now use a bounded background writer
rather than flushing from the motor loop. A 20-second preview after this change
completed at 20 Hz with no control gaps above 70 ms (zero motor writes). This
removes synchronous output as a source of control stalls; it does not establish
physical motion smoothness. Diagnostic records flush on clean exit.


### Fresh-response startup

Space first requests a fresh prediction while follower torque remains disabled.
The controller accepts a response only if it is no older than one replan interval
(0.5 seconds by default) and leaves enough horizon for another replan. Late
responses are logged and discarded; startup retries from a new observation before
enabling torque. During motion, late responses do not replace the existing chunk;
its original horizon-expiry stop still applies. The PNG encoder initializes before
Space, and client logs separate encoding/dispatch, HTTP, and response decoding.
An injected two-second delay was rejected in preview, followed by a fresh 72 ms
response and 160 control ticks with zero motor writes.

### Published models

Complete checkpoint packages are public, including tokenizer and normalization
artifacts. Repository loading instructions and training provenance are on the model cards:

- [SmolVLA](https://huggingface.co/vibecodemaster/white-lid-smolvla): main is 4k;
  branches `step-002000`, `step-004000`, and `step-005000`.
- [ACT](https://huggingface.co/vibecodemaster/white-lid-act): main is 20k;
  branches `step-005000`, `step-010000`, `step-015000`, and `step-020000`.

The Odin launchers still load their configured local checkpoint paths; publishing
these repositories does not automatically change which checkpoint a launcher selects.

### Arduino launcher

Run `./run_smolvla_ventuno_white_lid.sh` to use the Arduino-local runner through
ADB. It forwards terminal input and command arguments; compute stays on Arduino.
Set `SMOLVLA_ADB_SERIAL` for another device or `SMOLVLA_ARDUINO_RUNNER` for another
absolute runner path (`SMOLVLA_REMOTE_RUNNER` is also accepted). The default runner is
`/home/arduino/smolvla-white-lid/variants/192px-ab2-4/run_smolvla_white_lid.sh`.
