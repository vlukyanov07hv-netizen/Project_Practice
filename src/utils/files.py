"""Atomic outputs, relative project paths and reproducible run directories."""
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def replace_file(pending, path):
    delays = (.1, .2, .4, .8, 1., 1., 1., 1., 1.)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(pending, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


@contextmanager
def atomic_output(path, binary=False, encoding="utf-8", newline=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if binary else {"encoding": encoding, "newline": newline}
    with tempfile.NamedTemporaryFile(mode="w+b" if binary else "w", prefix=path.name + ".",
                                     suffix=".tmp", dir=str(path.parent), delete=False, **options) as stream:
        pending = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            try:
                pending.unlink()
            except OSError:
                pass
            raise
    replace_file(pending, path)


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_output(path) as stream:
        stream.write(text)


def write_text(path, value):
    with atomic_output(path) as stream:
        stream.write(value)


def save_torch(path, value, torch):
    with atomic_output(path, binary=True) as stream:
        torch.save(value, stream)


def relative(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def project_path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def new_run(root, name, mode):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = root / "results/runs" / (name + "_" + mode + "_" + stamp)
    path.mkdir(parents=True)
    return path
