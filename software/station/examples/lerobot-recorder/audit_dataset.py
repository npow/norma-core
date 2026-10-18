"""Read-only LeRobot audit: full video decode, table checks, and visual review sheets."""
import argparse
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from record import dataset_lock, quiet_video_logs


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit(root, output):
    quiet_video_logs()
    output.mkdir(parents=True, exist_ok=False)
    (output / "previews").mkdir()
    info = json.loads((root / "meta/info.json").read_text())
    rows = sorted(pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet"))).to_pylist(), key=lambda row: row["episode_index"])
    cameras = sorted(key for key, value in info["features"].items() if value["dtype"] == "video")
    manifest = {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
                for folder in ("data", "videos", "meta") for path in sorted((root / folder).rglob("*")) if path.is_file()}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    results, endpoints, all_states, all_actions = [], [], [], []
    referenced = set()
    total_decoded = 0
    global_index = 0
    for metadata in rows:
        index, n = metadata["episode_index"], metadata["length"]
        result = {"episode": index, "frames": n, "duration_s": n / info["fps"], "errors": [], "warnings": [], "cameras": {}}
        path = root / info["data_path"].format(chunk_index=metadata["data/chunk_index"], file_index=metadata["data/file_index"])
        referenced.add(str(path.relative_to(root)))
        table = pq.read_table(path).to_pydict()
        selected = np.asarray(table["episode_index"]) == index
        state = np.asarray(table["observation.state"], dtype=np.float32)[selected]
        action = np.asarray(table["action"], dtype=np.float32)[selected]
        all_states.append(state)
        all_actions.append(action)
        timestamps = np.asarray(table["timestamp"])[selected]
        def check(condition, message):
            if not condition:
                result["errors"].append(message)
        check(len(state) == n, "row count differs from episode metadata")
        check(state.shape == (n, 8) and action.shape == (n, 8), "joint/action shape differs from 8 motors")
        check(bool(np.isfinite(state).all() and np.isfinite(action).all()), "non-finite state/action values")
        check(bool(np.allclose(timestamps, np.arange(n) / info["fps"], atol=1e-4)), "irregular LeRobot timestamps")
        check(bool(np.array_equal(np.asarray(table["frame_index"])[selected], np.arange(n))), "frame index is not contiguous")
        check(bool(np.array_equal(np.asarray(table["index"])[selected], np.arange(global_index, global_index + n))), "global index is not contiguous")
        check(metadata["dataset_from_index"] == global_index and metadata["dataset_to_index"] == global_index + n, "episode boundaries disagree")
        check(bool(np.all(np.asarray(table["task_index"])[selected] == 0)), "unexpected task index")
        for name, values in (("state", state), ("action", action)):
            check(bool((values[:, :7] >= -100.001).all() and (values[:, :7] <= 100.001).all() and
                       (values[:, 7] >= -0.001).all() and (values[:, 7] <= 100.001).all()), f"{name} outside normalized calibration bounds")
        global_index += n
        if n:
            steps = np.abs(np.diff(state, axis=0))
            result.update({"state_span": np.ptp(state, axis=0).tolist(), "action_span": np.ptp(action, axis=0).tolist(),
                           "largest_joint_step": float(steps.max()) if n > 1 else 0,
                           "large_joint_steps_gt_20": int((steps > 20).any(axis=1).sum()) if n > 1 else 0,
                           "mean_action_state_error": float(np.abs(action - state).mean()),
                           "gripper_start_end": [float(state[0, -1]), float(state[-1, -1])]})
            if n < info["fps"] * 2:
                result["warnings"].append("shorter than 2 seconds")
            if np.max(np.ptp(state[:, :-1], axis=0)) < 3:
                result["warnings"].append("almost no arm movement (excluding gripper)")
            if result["large_joint_steps_gt_20"]:
                result["warnings"].append("joint jumps >20 normalized units per frame")
        timing = root / "meta/recording" / f"episode-{index:06d}.json"
        if timing.exists():
            times = np.asarray(json.loads(timing.read_text())["capture_time_ns"], dtype=np.int64)
            check(len(times) == n, "capture-time count differs from rows")
            gaps = np.diff(times) / 1e9
            result["capture_timing"] = {"max_gap_s": float(gaps.max()) if len(gaps) else 0,
                                        "gaps_gt_100ms": int((gaps > .1).sum()), "gaps_gt_250ms": int((gaps > .25).sum()),
                                        "actual_duration_s": float((times[-1] - times[0]) / 1e9) if len(times) else 0}
            if (gaps > .25).any():
                result["warnings"].append("capture gap >250ms")
        preview_indices = sorted(set(np.linspace(0, max(0, n - 1), 8, dtype=int).tolist()))
        samples = {}
        for camera in cameras:
            prefix = f"videos/{camera}"
            video = root / info["video_path"].format(video_key=camera, chunk_index=metadata[f"{prefix}/chunk_index"], file_index=metadata[f"{prefix}/file_index"])
            referenced.add(str(video.relative_to(root)))
            start, end = metadata[f"{prefix}/from_timestamp"], metadata[f"{prefix}/to_timestamp"]
            count, previous, duplicate_pairs, motion, brightness, contrasts, pts, longest_static, static_run = 0, None, 0, [], [], [], [], 0, 0
            samples[camera] = {}
            try:
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    stream.thread_count = 1
                    for frame in container.decode(stream):
                        when = float(frame.pts * frame.time_base)
                        if when < start - 1e-4 or when >= end - 1e-4:
                            continue
                        if count in preview_indices:
                            samples[camera][count] = frame.to_image()
                        small = np.asarray(frame.to_image().resize((64, 36)).convert("L"), dtype=np.float32)
                        brightness.append(float(small.mean()))
                        contrasts.append(float(small.std()))
                        if previous is not None:
                            change = float(np.abs(small - previous).mean())
                            motion.append(change)
                            duplicate_pairs += int(change == 0)
                            static_run = static_run + 1 if change < .1 else 0
                            longest_static = max(longest_static, static_run)
                        previous = small
                        pts.append(when - start)
                        check(frame.width == 398 and frame.height == 224, f"{camera}: unexpected image dimensions")
                        count += 1
                total_decoded += count
                check(count == n, f"{camera}: decoded {count} frames, expected {n}")
                check(bool(count == n and np.allclose(pts, np.arange(n) / info["fps"], atol=1e-4)), f"{camera}: irregular video PTS")
                metrics = {"decoded_frames": count, "mean_brightness": float(np.mean(brightness)),
                           "min_spatial_contrast": float(min(contrasts, default=0)),
                           "near_blank_frames": sum(c < 2 for c in contrasts),
                           "duplicate_adjacent_frames": duplicate_pairs,
                           "longest_near_static_s": longest_static / info["fps"],
                           "mean_pixel_motion": float(np.mean(motion)) if motion else 0}
                result["cameras"][camera] = metrics
                if metrics["near_blank_frames"]:
                    result["warnings"].append(f"{camera}: nearly blank frames")
                if metrics["longest_near_static_s"] > 3:
                    result["warnings"].append(f"{camera}: near-static sequence >3s (review)")
            except Exception as exc:
                result["errors"].append(f"{camera}: decode failed: {exc}")
        sheet = Image.new("RGB", (398 * len(preview_indices), 249 * len(cameras)), "#171a20")
        draw = ImageDraw.Draw(sheet)
        for y, camera in enumerate(cameras):
            for x, frame_index in enumerate(preview_indices):
                if frame_index in samples[camera]:
                    sheet.paste(samples[camera][frame_index], (398 * x, 249 * y + 25))
                draw.text((398 * x + 4, 249 * y + 5), f"ep {index:03d} {camera.split('.')[-1]} t={frame_index / info['fps']:.2f}s", fill="white")
        sheet.save(output / "previews" / f"episode-{index:06d}.jpg", quality=88)
        endpoints.append((index, [samples[cameras[0]].get(0), samples[cameras[0]].get(n - 1), samples[cameras[1]].get(n - 1)]))
        results.append(result)
        print(f"Episode {index:03d}: {n} rows, {len(result['errors'])} errors, {result['warnings']}", flush=True)
    for page_start in range(0, len(endpoints), 8):
        batch = endpoints[page_start:page_start + 8]
        sheet = Image.new("RGB", (398 * 3, len(batch) * 250), "#171a20")
        draw = ImageDraw.Draw(sheet)
        for y, (index, frames) in enumerate(batch):
            for x, image in enumerate(frames):
                if image is not None:
                    sheet.paste(image, (398 * x, y * 250 + 25))
                draw.text((398 * x + 4, y * 250 + 5), f"EP {index:03d} | {['start camera_0', 'end camera_0', 'end camera_1'][x]}", fill="white")
        sheet.save(output / f"endpoints-{page_start // 8:02d}.jpg", quality=90)
    actual_files = {str(p.relative_to(root)) for folder, extension in (("data", "*.parquet"), ("videos", "*.mp4")) for p in (root / folder).rglob(extension)}
    source_unchanged = all((root / path).exists() and digest(root / path) == entry["sha256"] for path, entry in manifest.items())
    summary = {"source": str(root), "generated_at": datetime.datetime.now().isoformat(),
               "episodes": len(results), "frames": sum(r["frames"] for r in results),
               "expected_camera_frames": sum(r["frames"] for r in results) * len(cameras), "decoded_camera_frames": total_decoded,
               "episodes_with_errors": [r["episode"] for r in results if r["errors"]],
               "episodes_with_warnings": [r["episode"] for r in results if r["warnings"]],
               "timing_audited_episodes": sum("capture_timing" in r for r in results),
               "missing_files": sorted(referenced - actual_files), "unindexed_files": sorted(actual_files - referenced),
               "source_hashes_unchanged": source_unchanged,
               "state_min": np.concatenate(all_states).min(axis=0).tolist(), "state_max": np.concatenate(all_states).max(axis=0).tolist(),
               "action_min": np.concatenate(all_actions).min(axis=0).tolist(), "action_max": np.concatenate(all_actions).max(axis=0).tolist()}
    assert summary["episodes"] == info["total_episodes"] and summary["frames"] == info["total_frames"]
    (output / "audit.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    columns = ["episode", "frames", "duration_s", "largest_joint_step", "large_joint_steps_gt_20", "mean_action_state_error", "errors", "warnings"]
    with (output / "episodes.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    cards = []
    for row in results:
        i = row["episode"]
        videos = []
        meta = rows[i]
        for camera in cameras:
            prefix = f"videos/{camera}"
            relative = Path("..") / root.name / info["video_path"].format(video_key=camera, chunk_index=meta[f"{prefix}/chunk_index"], file_index=meta[f"{prefix}/file_index"])
            videos.append(f'<video controls preload="none" src="{html.escape(str(relative))}"></video>')
        flags = "; ".join(row["errors"] + row["warnings"]) or "No structural flags"
        cards.append(f'<article id="episode-{i}"><h2>Episode {i:03d} · {row["duration_s"]:.2f}s · {row["frames"]} frames</h2><p>{html.escape(flags)}</p><div>{"".join(videos)}</div><a href="previews/episode-{i:06d}.jpg"><img loading="lazy" src="previews/episode-{i:06d}.jpg"></a></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>White lid dataset review</title><style>body{font:16px system-ui;background:#10151c;color:#eef;padding:24px;max-width:1300px;margin:auto}article{padding:20px;border:1px solid #456;margin:24px 0}video{width:49%}img{width:100%}a{color:#9de}</style><h1>White lid: episode review</h1><p>Both camera videos are shown for every episode. Click the timeline to enlarge it. Source data has not been modified.</p>' + "".join(cards))
    (output / "white-lid_eda_report.md").write_text(
        f"# White lid dataset audit\n\nGenerated: {summary['generated_at']}\n\nSource: `{root}`\n\n"
        "## Basic information and file format\n\nLeRobot v3; Apache Parquet numeric tables, H.264 MP4 camera streams, JSON metadata. "
        "PyArrow, NumPy, and PyAV were used for a full local integrity audit.\n\n"
        "## Data analysis\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"
        "## Key findings and quality assessment\n\nSee `episodes.csv` for per-episode flags and `audit.json` for complete metrics. "
        "Every table row and video frame was examined for structural integrity; visual task success requires reviewing the previews and videos. "
        "Nominal LeRobot/video timestamps do not establish actual acquisition timing for episodes without capture-time sidecars.\n\n"
        "## Recommendations\n\nUse `index.html` to review task success. Static frames and motion thresholds are review flags, not proof of failure. "
        "Do not remove episodes solely for these heuristic flags. Retain the frozen `source_manifest.json` when curating.\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"REVIEW={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output or root.parent / f"{root.name}-review-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    with dataset_lock(root):
        audit(root, output)
