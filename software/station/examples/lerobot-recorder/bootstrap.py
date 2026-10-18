"""Generate private protobuf bindings without changing the worktree's target/."""
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def setup():
    sources = sorted((REPO / "protobufs").rglob("*.proto"))
    sources += sorted((REPO / "shared/gremlin_py").rglob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    cache = Path.home() / ".cache/norma-recorder" / digest.hexdigest()[:16]
    if not (cache / "ready").exists():
        for directory in sorted((REPO / "protobufs").iterdir()):
            if directory.is_dir():
                subprocess.run([
                    sys.executable, str(REPO / "shared/gremlin_py/gremlin.py"),
                    "--proto-root", str(directory),
                    "--target-root", str(cache / "target/gen_python/protobuf" / directory.name),
                    "--project-root", str(cache),
                    "--gremlin-import-path", "shared.gremlin_py.gremlin",
                ], check=True, capture_output=True)
        (cache / "ready").touch()
    sys.path[:0] = [str(cache), str(REPO), str(REPO / "software/station/examples/st3215-remote-teleop-py")]
