#!/usr/bin/env python3
"""Export contiguous human corrections as ordinary LeRobot training episodes."""
import argparse
import hashlib
import json
from pathlib import Path


def selected_segments(owners, times, fps, minimum=1):
    """Split at ownership changes and timing gaps; never label policy actions human."""
    start = None
    for i in range(len(owners) + 1):
        selected = i < len(owners) and owners[i] == 1
        gap = i > 0 and i < len(times) and not 0 < times[i] - times[i - 1] <= 1.5 / fps
        if start is not None and (not selected or gap):
            if i - start >= minimum:
                yield start, i
            start = None
        if selected and start is None:
            start = i


def export(source_root, output_root, minimum=1, repo_id="local/human-corrections"):
    from record import dataset_lock
    with dataset_lock(Path(source_root)):
        return _export(source_root, output_root, minimum, repo_id)


def _export(source_root, output_root, minimum, repo_id):
    from bootstrap import setup
    setup()
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from record import quiet_video_logs, separate_episode_files
    quiet_video_logs()
    source_root, output_root = Path(source_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if minimum < 1:
        raise ValueError("minimum must be positive")
    if output_root.exists():
        raise ValueError("Output must be a new directory; existing data is preserved")
    info = json.loads((source_root / "meta/info.json").read_text())
    source = LeRobotDataset(info.get("repo_id", "local/rollout-interventions"), root=source_root, video_backend="pyav")
    features = {key: {field: value for field, value in feature.items() if field in ("dtype", "shape", "names")}
                for key, feature in source.features.items()
                if key in ("observation.state", "action") or key.startswith("observation.images.")}
    if "intervention.owner" not in source.features:
        raise ValueError("Source is not an intervention dataset")
    segments, source_hashes = [], {}
    for episode in source.meta.episodes:
        index = int(episode["episode_index"])
        base = source_root / "meta/interventions" / f"episode-{index:06d}"
        journal = base.with_suffix(".jsonl")
        audit_path = base.with_suffix(".json")
        if not journal.is_file() or not audit_path.is_file():
            raise ValueError(f"Episode {index}: missing audit/journal; review interrupted capture before export")
        audit = json.loads(audit_path.read_text())
        for relative, expected_hash in audit["file_sha256"].items():
            path = (source_root / relative).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError("Audit path escapes dataset root")
            with path.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Episode {index}: checksum mismatch for {relative}")
        raw = journal.read_bytes()
        details = [json.loads(line) for line in raw.splitlines()]
        audit = json.loads(audit_path.read_text())
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        if len(details) != end - start or audit["frames"] != end - start:
            raise ValueError(f"Episode {index}: frame count audit failed")
        owners = [int(source.hf_dataset[i]["intervention.owner"].item()) for i in range(start, end)]
        expected = [{"policy": 0, "human": 1, "transition": 2}[row["owner"]] for row in details]
        if owners != expected or [row["frame_index"] for row in details] != list(range(end - start)):
            raise ValueError(f"Episode {index}: ownership/index audit failed")
        source_hashes[str(index)] = {"journal_sha256": hashlib.sha256(raw).hexdigest(),
            "capture_frame_sha256": audit["frame_sha256"], "frames": len(details)}
        for lo, hi in selected_segments(owners, [row["time_s"] for row in details], source.fps, minimum):
            segments.append((index, start + lo, start + hi))
    if not segments:
        raise ValueError("No human correction segments meet --min-frames")
    output = LeRobotDataset.create(repo_id=repo_id, root=output_root, fps=source.fps,
        robot_type=source.meta.robot_type, features=features, use_videos=True, vcodec="h264", image_writer_threads=2)
    separate_episode_files(output)
    manifest = {"source": str(source_root), "selection": "human only; transitions excluded; timing gaps split",
        "minimum_frames": minimum, "source_audits": source_hashes, "segments": []}
    try:
        for source_episode, lo, hi in segments:
            digest = hashlib.sha256()
            for index in range(lo, hi):
                row = source[index]
                frame = {"task": row["task"]}
                for key, feature in features.items():
                    value = row[key].numpy()
                    if feature["dtype"] == "video":
                        value = np.rint(np.clip(value.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    frame[key] = value
                    digest.update(key.encode())
                    digest.update(value.tobytes())
                output.add_frame(frame)
            output.save_episode()
            manifest["segments"].append({"episode_index": output.meta.total_episodes - 1,
                "source_episode": source_episode, "source_from_index": lo, "source_to_index": hi,
                "frames": hi - lo, "decoded_frame_sha256": digest.hexdigest()})
    finally:
        output.finalize()
        output.stop_image_writer()
    manifest["total_frames"] = sum(segment["frames"] for segment in manifest["segments"])
    manifest["total_episodes"] = len(manifest["segments"])
    collection = json.loads((source_root / "meta/collection.json").read_text())
    (output_root / "meta/collection.json").write_text(json.dumps(collection, indent=2) + "\n")
    (output_root / "meta/intervention-export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {manifest['total_frames']} human frames in {manifest['total_episodes']} episodes to {output_root}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=1, help="Prefer at least one training action chunk")
    parser.add_argument("--repo-id", default="local/human-corrections")
    args = parser.parse_args()
    if args.min_frames < 1:
        parser.error("--min-frames must be positive")
    export(args.source, args.output, args.min_frames, args.repo_id)


if __name__ == "__main__":
    main()
